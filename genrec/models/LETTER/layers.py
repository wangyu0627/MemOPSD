from types import SimpleNamespace
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.sklearn_threading import limit_sklearn_threads


class MLP(nn.Module):
    def __init__(self, hidden_sizes: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_size, output_size))
            if idx != len(hidden_sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LETTERQuantizationLayer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        latent_dim: int,
        commitment_weight: float,
        diversity_temperature: float,
        kmeans_num_threads: int = None,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_weight = commitment_weight
        self.diversity_temperature = diversity_temperature
        self.kmeans_num_threads = kmeans_num_threads
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / max(codebook_size, 1),
            1.0 / max(codebook_size, 1),
        )

    def initialize_codebook(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.size(0) == 0:
            return residual

        n_clusters = min(self.codebook_size, residual.size(0))
        residual_cpu = residual.detach().cpu().numpy()
        if n_clusters > 1:
            with limit_sklearn_threads(self.kmeans_num_threads):
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(residual_cpu)
            centers = torch.tensor(
                kmeans.cluster_centers_,
                dtype=residual.dtype,
                device=residual.device,
            )
        else:
            labels = torch.zeros(residual.size(0), dtype=torch.long).numpy()
            centers = residual[:1].detach()

        if n_clusters < self.codebook_size:
            sampled = torch.randint(
                0,
                residual.size(0),
                (self.codebook_size - n_clusters,),
                device=residual.device,
            )
            centers = torch.cat([centers, residual[sampled].detach()], dim=0)
        self.embedding.weight.data.copy_(centers[: self.codebook_size])
        return residual - self.embedding(torch.tensor(labels, device=residual.device))

    def forward(self, residual: torch.Tensor):
        distances = (
            residual.pow(2).sum(dim=1, keepdim=True)
            - 2 * residual @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = torch.argmin(distances, dim=-1)
        quantized = self.embedding(indices)

        codebook_loss = F.mse_loss(quantized, residual.detach())
        commitment_loss = F.mse_loss(quantized.detach(), residual)
        quant_loss = codebook_loss + self.commitment_weight * commitment_loss

        probs = F.softmax(
            -distances / max(self.diversity_temperature, 1e-6),
            dim=-1,
        )
        avg_probs = probs.mean(dim=0)
        diversity_loss = torch.sum(
            avg_probs * torch.log(avg_probs * self.codebook_size + 1e-8)
        )

        quantized = residual + (quantized - residual).detach()
        return quantized, quant_loss, diversity_loss, indices

    @torch.no_grad()
    def encode(self, residual: torch.Tensor):
        distances = (
            residual.pow(2).sum(dim=1, keepdim=True)
            - 2 * residual @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = torch.argmin(distances, dim=-1)
        return self.embedding(indices), indices


class LETTERRQVAEModel(nn.Module):
    def __init__(
        self,
        hidden_sizes: list[int],
        n_codebooks: int,
        codebook_size: Union[int, list[int]],
        latent_dim: int,
        dropout: float,
        quant_loss_weight: float,
        commitment_weight: float,
        cf_weight: float,
        diversity_weight: float,
        diversity_temperature: float,
        kmeans_num_threads: int = None,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        if isinstance(codebook_size, int):
            self.codebook_sizes = [codebook_size] * n_codebooks
        else:
            self.codebook_sizes = codebook_size
        self.quant_loss_weight = quant_loss_weight
        self.cf_weight = cf_weight
        self.diversity_weight = diversity_weight

        self.encoder = MLP(hidden_sizes + [latent_dim], dropout)
        self.quantization_layers = nn.ModuleList(
            [
                LETTERQuantizationLayer(
                    codebook_size=size,
                    latent_dim=latent_dim,
                    commitment_weight=commitment_weight,
                    diversity_temperature=diversity_temperature,
                    kmeans_num_threads=kmeans_num_threads,
                )
                for size in self.codebook_sizes
            ]
        )
        self.decoder = MLP([latent_dim] + hidden_sizes[::-1], dropout)

    @torch.no_grad()
    def generate_codebook(self, x: torch.Tensor):
        residual = self.encoder(x)
        for layer in self.quantization_layers:
            residual = layer.initialize_codebook(residual)

    def _quantize(self, encoded: torch.Tensor):
        residual = encoded
        quantized_sum = torch.zeros_like(encoded)
        indices = []
        quant_losses = []
        diversity_losses = []
        for layer in self.quantization_layers:
            quantized, quant_loss, diversity_loss, index = layer(residual)
            residual = residual - quantized
            quantized_sum = quantized_sum + quantized
            indices.append(index)
            quant_losses.append(quant_loss)
            diversity_losses.append(diversity_loss)

        return (
            quantized_sum,
            torch.stack(quant_losses).mean(),
            torch.stack(diversity_losses).mean(),
            torch.stack(indices, dim=-1),
        )

    def _cf_loss(self, quantized: torch.Tensor, cf_embeddings: torch.Tensor):
        if cf_embeddings is None:
            return quantized.sum() * 0.0
        quantized = F.normalize(quantized, dim=-1)
        cf_embeddings = F.normalize(cf_embeddings, dim=-1)
        labels = torch.arange(quantized.size(0), device=quantized.device)
        logits = torch.matmul(quantized, cf_embeddings.t())
        return F.cross_entropy(logits, labels)

    def forward(self, x: torch.Tensor, cf_embeddings: torch.Tensor = None):
        encoded = self.encoder(x)
        quantized, quant_loss, diversity_loss, indices = self._quantize(encoded)
        decoded = self.decoder(quantized)
        recon_loss = F.mse_loss(decoded, x, reduction="mean")
        cf_loss = self._cf_loss(quantized, cf_embeddings)
        loss = (
            recon_loss
            + self.quant_loss_weight * quant_loss
            + self.cf_weight * cf_loss
            + self.diversity_weight * diversity_loss
        )
        return SimpleNamespace(
            loss=loss,
            recon_loss=recon_loss,
            quant_loss=quant_loss,
            cf_loss=cf_loss,
            diversity_loss=diversity_loss,
            indices=indices,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        encoded = self.encoder(x)
        residual = encoded
        indices = []
        for layer in self.quantization_layers:
            quantized, index = layer.encode(residual)
            residual = residual - quantized
            indices.append(index)
        return torch.stack(indices, dim=-1).detach().cpu().numpy()
