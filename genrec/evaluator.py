import torch


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k,
            'err': self.err_at_k
        }

        self.legal_preds = set()
        legal_pred_tokens = []
        for iid, tokens in self.tokenizer.item2tokens.items():
            if isinstance(tokens, int):
                legal_pred = (tokens,)
            else:
                legal_pred = tuple(tokens)
            self.legal_preds.add(legal_pred)
            legal_pred_tokens.append(legal_pred)

        self.legal_pred_tensor = None
        self.legal_pred_len = None
        legal_pred_lens = {len(tokens) for tokens in legal_pred_tokens}
        if len(legal_pred_lens) == 1 and legal_pred_tokens:
            self.legal_pred_len = legal_pred_lens.pop()
            self.legal_pred_tensor = torch.tensor(legal_pred_tokens, dtype=torch.long)

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

    def _normalize_topk(self, topk=None):
        if topk is None:
            topk = self.config['topk']
        if isinstance(topk, int):
            return [topk]
        return list(topk)

    def calculate_pos_index(self, preds, labels, topk=None):
        maxk = max(self._normalize_topk(topk))
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] >= maxk, f"preds.shape[1] = {preds.shape[1]} < {maxk}"

        pred_len = preds.shape[-1]
        if labels.shape[-1] < pred_len:
            return self._calculate_pos_index_loop(preds, labels, maxk)

        labels_for_match = labels[:, :pred_len]
        label_lengths = torch.full(
            (labels.shape[0],),
            labels.shape[-1],
            dtype=torch.long,
        )
        if self.eos_token is not None:
            eos_matches = labels.eq(self.eos_token)
            has_eos = eos_matches.any(dim=1)
            eos_pos = eos_matches.float().argmax(dim=1).long()
            label_lengths = torch.where(has_eos, eos_pos, label_lengths)

        length_matches = label_lengths.eq(pred_len).unsqueeze(1)
        token_matches = preds[:, :maxk, :].eq(labels_for_match.unsqueeze(1)).all(dim=-1)
        matches = token_matches & length_matches

        # Preserve the original one-ground-truth behavior: duplicate correct
        # predictions after the first hit do not increase recall.
        return matches & matches.cumsum(dim=1).eq(1)

    def _calculate_pos_index_loop(self, preds, labels, maxk):
        pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def err_at_k(self, preds, k):
        """
        Calculate the percentage illegal predictions
        among the top k generated token sequences.
        """
        encoded = self._encoded_legal_mask(preds[:, :k])
        if encoded is not None:
            return (~encoded).float().mean(dim=1).cpu()

        ret = []
        for i in range(preds.shape[0]):
            n_illegal_preds = 0
            for j in range(k):
                cur_pred = tuple(preds[i, j].tolist())
                if cur_pred not in self.legal_preds:
                    n_illegal_preds += 1
            ret.append(n_illegal_preds / k)
        return torch.FloatTensor(ret)

    def _encoded_legal_mask(self, preds):
        if (
            self.legal_pred_tensor is None
            or self.legal_pred_len is None
            or preds.shape[-1] != self.legal_pred_len
            or not hasattr(torch, 'isin')
        ):
            return None

        preds = preds.detach().cpu().long()
        legal_preds = self.legal_pred_tensor
        max_token = max(
            int(preds.max().item()) if preds.numel() else 0,
            int(legal_preds.max().item()) if legal_preds.numel() else 0,
        )
        base = max_token + 1
        if base <= 1:
            base = 2

        max_encoded = base ** self.legal_pred_len - 1
        if max_encoded > torch.iinfo(torch.long).max:
            return None

        powers = torch.tensor(
            [base ** i for i in range(self.legal_pred_len)],
            dtype=torch.long,
        )
        legal_keys = (legal_preds.long() * powers).sum(dim=-1).unique()
        pred_keys = (preds * powers).sum(dim=-1)
        return torch.isin(pred_keys, legal_keys)

    def calculate_metrics(self, preds, labels, topk=None):
        topk = self._normalize_topk(topk)
        results = {}
        pos_index = self.calculate_pos_index(preds, labels, topk=topk)
        for metric in self.config['metrics']:
            for k in topk:
                if metric in ['err']:
                    results[f"{metric}@{k}"] = self.metric2func[metric](preds, k)
                else:
                    results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        return results
