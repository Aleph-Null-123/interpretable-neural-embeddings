import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardSAE(nn.Module):
    def __init__(self, input_dim, expansion_factor=2, sparsity_lambda=1e-3):
        super().__init__()
        self.latent_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        recon = self.decoder(latent)
        l1_penalty = self.sparsity_lambda * torch.norm(latent, p=1)
        return recon, latent, l1_penalty


class TopKSAE(nn.Module):
    def __init__(self, input_dim, expansion_factor=2, topk=32):
        super().__init__()
        self.latent_dim = input_dim * expansion_factor
        self.topk = topk
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim)

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        topk_vals, _ = torch.topk(latent, min(self.topk, self.latent_dim), dim=1)
        threshold = topk_vals[:, -1].unsqueeze(1)
        sparse_latent = latent * (latent >= threshold)
        recon = self.decoder(sparse_latent)
        return recon, sparse_latent, 0.0

class JumpReLUSAE(nn.Module):
    def __init__(self, input_dim, expansion_factor=2, bandwidth=1e-3, sparsity_lambda=1e-3):
        super().__init__()
        self.latent_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim)
        self.bandwidth = bandwidth
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        jump_penalty = self.sparsity_lambda * torch.sum(1.0 - torch.exp(-latent / self.bandwidth))
        recon = self.decoder(latent)
        return recon, latent, jump_penalty


class GatedSAE(nn.Module):  # wont use for this project
    def __init__(self, input_dim, expansion_factor=2, sparsity_lambda=1e-3):
        super().__init__()
        self.latent_dim = input_dim * expansion_factor
        self.encoder = nn.Linear(input_dim, self.latent_dim)
        self.gate = nn.Linear(input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, input_dim)
        self.sparsity_lambda = sparsity_lambda

    def forward(self, x):
        latent = F.relu(self.encoder(x))
        gated = latent * torch.sigmoid(self.gate(x))
        recon = self.decoder(gated)
        l1_penalty = self.sparsity_lambda * torch.norm(gated, p=1)
        return recon, gated, l1_penalty