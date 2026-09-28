import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from genrec.dataset import AbstractDataset
from genrec.models.LETTER.layers import LETTERRQVAEModel
from genrec.models.TIGER.tokenizer import TIGERTokenizer
from genrec.sklearn_threading import limit_sklearn_threads
from genrec.utils import list_to_str


class LETTERTokenizer(TIGERTokenizer):
    """LETTER learnable tokenizer adapted to the genrec TIGER-style interface."""

    def _init_tokenizer(self, dataset: AbstractDataset):
        self.sem_ids_path = self.config.get(
            "sem_ids_path",
            os.path.join(
                dataset.cache_dir,
                "processed",
                "letter_"
                f"{os.path.basename(self.config['sent_emb_model'])}_"
                f"{list_to_str(self.codebook_sizes, remove_blank=True)}.sem_ids",
            ),
        )

        if self.use_atomic_ids:
            if not os.path.exists(self.sem_ids_path):
                item2sem_ids = self._generate_atomic_ids(dataset)
            else:
                item2sem_ids = json.load(open(self.sem_ids_path, "r"))
            return self._sem_ids_to_tokens(item2sem_ids)

        if not os.path.exists(self.sem_ids_path):
            sent_embs = self._load_or_encode_sentence_embeddings(dataset)
            cf_embs = self._build_cf_embeddings(dataset, self.config["letter_latent_dim"])
            train_mask = self._get_items_for_training(dataset)

            model_path = os.path.join(
                dataset.cache_dir,
                "processed",
                "letter_rqvae_"
                f"{list_to_str(self.codebook_sizes, remove_blank=True)}.pth",
            )
            rqvae_model = self._train_letter_rqvae(
                sent_embs=torch.FloatTensor(sent_embs[train_mask]).to(
                    self.config["device"]
                ),
                cf_embs=torch.FloatTensor(cf_embs[train_mask]).to(
                    self.config["device"]
                ),
                model_path=model_path,
            )
            self._generate_letter_semantic_id(
                rqvae_model=rqvae_model,
                sent_embs=torch.FloatTensor(sent_embs).to(self.config["device"]),
                sem_ids_path=self.sem_ids_path,
            )

        self.log(f"[TOKENIZER] Loading LETTER semantic IDs from {self.sem_ids_path}...")
        item2sem_ids = json.load(open(self.sem_ids_path, "r"))
        return self._sem_ids_to_tokens(item2sem_ids)

    def _load_or_encode_sentence_embeddings(self, dataset: AbstractDataset):
        assert self.config["metadata"] == "sentence", (
            "LETTERTokenizer needs sentence metadata for learnable item tokenization."
        )
        sent_emb_path = os.path.join(
            dataset.cache_dir,
            "processed",
            f"{os.path.basename(self.config['sent_emb_model'])}.sent_emb",
        )
        if os.path.exists(sent_emb_path):
            self.log(f"[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...")
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
                -1,
                self.config["sent_emb_dim"],
            )
        else:
            self.log("[TOKENIZER] Encoding sentence embeddings for LETTER...")
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

        if self.config["sent_emb_pca"] > 0:
            self.log("[TOKENIZER] Applying PCA to sentence embeddings...")
            with limit_sklearn_threads(self.config.get("letter_sklearn_num_threads")):
                from sklearn.decomposition import PCA

                pca = PCA(n_components=self.config["sent_emb_pca"], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
        self.log(f"[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}")
        return sent_embs.astype(np.float32)

    def _build_cf_embeddings(self, dataset: AbstractDataset, target_dim: int):
        n_items = dataset.n_items - 1
        if n_items <= 0:
            return np.zeros((0, target_dim), dtype=np.float32)

        rows, cols, data = [], [], []
        window = self.config["letter_cf_window"]
        for item_seq in dataset.split_data["train"]["item_seq"]:
            item_ids = [
                dataset.item2id[item] - 1
                for item in item_seq
                if item in dataset.item2id and dataset.item2id[item] > 0
            ]
            for pos, item_id in enumerate(item_ids):
                start = max(0, pos - window)
                end = min(len(item_ids), pos + window + 1)
                for ctx_pos in range(start, end):
                    if ctx_pos == pos:
                        continue
                    rows.append(item_id)
                    cols.append(item_ids[ctx_pos])
                    data.append(1.0)

        if not data:
            rng = np.random.default_rng(self.config["rand_seed"])
            return rng.normal(0.0, 0.01, size=(n_items, target_dim)).astype(np.float32)

        n_components = min(target_dim, max(1, n_items - 1))
        with limit_sklearn_threads(self.config.get("letter_sklearn_num_threads")):
            from scipy.sparse import coo_matrix, identity
            from sklearn.decomposition import TruncatedSVD

            matrix = coo_matrix((data, (rows, cols)), shape=(n_items, n_items)).tocsr()
            matrix = matrix + identity(n_items, format="csr")
            svd = TruncatedSVD(
                n_components=n_components,
                random_state=self.config["rand_seed"],
            )
            cf_embs = svd.fit_transform(matrix).astype(np.float32)
        if n_components < target_dim:
            cf_embs = np.pad(cf_embs, ((0, 0), (0, target_dim - n_components)))
        norms = np.linalg.norm(cf_embs, axis=1, keepdims=True)
        return cf_embs / np.maximum(norms, 1e-12)

    def _train_letter_rqvae(
        self,
        sent_embs: torch.Tensor,
        cf_embs: torch.Tensor,
        model_path: str,
    ):
        device = self.config["device"]
        hidden_sizes = [sent_embs.shape[1]] + self.config["letter_tokenizer_hidden_sizes"]
        rqvae_model = LETTERRQVAEModel(
            hidden_sizes=hidden_sizes,
            n_codebooks=self.config["rq_n_codebooks"],
            codebook_size=self.config["rq_codebook_size"],
            latent_dim=self.config["letter_latent_dim"],
            dropout=self.config["letter_tokenizer_dropout"],
            quant_loss_weight=self.config["letter_quant_loss_weight"],
            commitment_weight=self.config["letter_commitment_weight"],
            cf_weight=self.config["letter_cf_weight"],
            diversity_weight=self.config["letter_diversity_weight"],
            diversity_temperature=self.config["letter_diversity_temperature"],
            kmeans_num_threads=self.config.get("letter_sklearn_num_threads"),
        ).to(device)

        self.log(rqvae_model)
        if os.path.exists(model_path):
            self.log(f"[TOKENIZER] Loading LETTER RQ-VAE model from {model_path}...")
            rqvae_model.load_state_dict(torch.load(model_path, map_location=device))
            return rqvae_model

        if self.config["letter_kmeans_init"]:
            self.log("[TOKENIZER] Initializing LETTER codebooks with KMeans...")
            rqvae_model.generate_codebook(sent_embs)

        dataset = TensorDataset(sent_embs, cf_embs)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config["letter_tokenizer_batch_size"],
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            rqvae_model.parameters(),
            lr=self.config["letter_tokenizer_lr"],
            weight_decay=self.config["letter_tokenizer_weight_decay"],
        )

        epochs = self.config["letter_tokenizer_epochs"]
        self.log("[TOKENIZER] Training LETTER RQ-VAE tokenizer...")
        rqvae_model.train()
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            for x_batch, cf_batch in dataloader:
                optimizer.zero_grad()
                outputs = rqvae_model(x_batch, cf_batch)
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.detach().cpu().item()
            if (epoch + 1) % max(1, epochs // 10) == 0:
                self.log(
                    f"[TOKENIZER] LETTER RQ-VAE epoch {epoch + 1}/{epochs}, "
                    f"loss={total_loss / len(dataloader):.6f}"
                )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(rqvae_model.state_dict(), model_path, pickle_protocol=4)
        return rqvae_model

    def _generate_letter_semantic_id(
        self,
        rqvae_model: LETTERRQVAEModel,
        sent_embs: torch.Tensor,
        sem_ids_path: str,
    ):
        rqvae_model.eval()
        letter_sem_ids = rqvae_model.encode(sent_embs)
        item2sem_ids = self._extend_semantic_ids(letter_sem_ids)
        self.log(f"[TOKENIZER] Saving LETTER semantic IDs to {sem_ids_path}...")
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, "w") as f:
            json.dump(item2sem_ids, f)
