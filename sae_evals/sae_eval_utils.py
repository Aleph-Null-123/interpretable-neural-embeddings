import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sae_architectures import *

def get_model(arch, input_dim, exp, l1=1e-3, topk=None, bw=None):
    if arch == "standard":
        return StandardSAE(input_dim, expansion_factor=exp, sparsity_lambda=l1)
    if arch == "topk":
        return TopKSAE(input_dim, expansion_factor=exp, topk=topk)
    if arch == "jumprelu":
        return JumpReLUSAE(input_dim, expansion_factor=exp, bandwidth=bw, sparsity_lambda=l1)
    if arch == "gated":
        return GatedSAE(input_dim, expansion_factor=exp, sparsity_lambda=l1)
    raise ValueError(f"Unknown architecture type: {arch}")
    
def load_sae_and_embeddings(model_path, emb_path, arch, input_dim, expansion_factor, l1=0.0, topk=None, bw=None):
    model = get_model(arch, input_dim, expansion_factor, l1=l1, topk=topk, bw=bw)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    emb = np.load(emb_path)
    data_tensor = torch.tensor(emb).float()

    with torch.no_grad():
        recon, latents, _ = model(data_tensor)

    return emb, latents.numpy(), recon.numpy()


def compute_explained_variance(original, reconstructed):
    return r2_score(original, reconstructed, multioutput='uniform_average')


def compute_l0_sparsity(latents):
    return np.mean(np.sum(latents != 0, axis=1) == 0)


def get_nonzero_ratio(latents):
    return np.mean(np.count_nonzero(latents, axis=1) / latents.shape[1])


def plot_sparsity_vs_variance(sparsities, variances, labels):
    plt.figure()
    for s, v, label in zip(sparsities, variances, labels):
        plt.scatter(s, v, label=label)
    plt.xlabel("Nonzero Ratio")
    plt.ylabel("Explained Variance (R^2)")
    plt.title("Sparsity vs. Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_histogram(latents, bins=50):
    plt.figure()
    plt.hist(latents.flatten(), bins=bins, log=True)
    plt.xlabel("Latent Activation")
    plt.ylabel("Frequency")
    plt.title("Histogram of Latent Activations")
    plt.tight_layout()
    plt.show()
