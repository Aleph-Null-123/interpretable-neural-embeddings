from sae_architectures2 import TopKSAE as TopKSAE
import numpy as np
from sklearn.linear_model import RidgeClassifier, RidgeCV
from sklearn.metrics import r2_score, accuracy_score
from scipy.stats import pearsonr
import os
import torch
from sae_architectures2 import TopKSAE as TopKSAE

def get_topk_latents_by_params(name,dim,embedding_array,topk,expansion_factor=8,
                               lr=1e-3,fixed_l1="0.0001",root_model_dir="../sae_gridsearch_2_fixed",device='cpu'):
    model = TopKSAE(input_dim=dim, expansion_factor=expansion_factor, topk=topk)
    model_name = f"topk_exp{expansion_factor}_lr{lr}_l1{fixed_l1}_topk{topk}_model.pt"
    model_path = os.path.join(root_model_dir, f"{name}_{dim}", model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        data_tensor = torch.tensor(embedding_array).float().to(device)
        _, latents, _ = model(data_tensor)

    return latents.cpu().numpy()


def compute_position_r2(z, position):
    model = RidgeCV(alphas=np.logspace(-6, 6, 13))
    model.fit(z, position)
    pred = model.predict(z)
    return r2_score(position, pred)


def compute_direction_acc(z, direction):
    model = RidgeClassifier(alpha=1.0)
    model.fit(z, direction)
    pred = model.predict(z)
    return accuracy_score(direction, pred)


def compute_behavior_corr(z, behavior):
    correlations = np.zeros((z.shape[1], behavior.shape[1]))
    for i in range(z.shape[1]):
        for j in range(behavior.shape[1]):
            try:
                corr, _ = pearsonr(z[:, i], behavior[:, j])
            except Exception:
                corr = np.nan
            correlations[i, j] = corr
    return correlations


def compute_sparsity(z):
    return 100 * (np.count_nonzero(z, axis=1).mean() / z.shape[1])


def neuron_latent_attribution(neural, latents):
    correlations = np.zeros((neural.shape[1], latents.shape[1]))
    for i in range(neural.shape[1]):
        for j in range(latents.shape[1]):
            try:
                corr, _ = pearsonr(neural[:, i], latents[:, j])
            except Exception:
                corr = np.nan
            correlations[i, j] = corr
    return correlations


def compute_score(corr, k=10):
    corr = np.abs(corr)
    corr = np.nan_to_num(corr, nan=0.0)
    flat_corr = np.abs(corr).flatten()
    top_k = np.partition(flat_corr, -k)[-k:]
    return np.mean(top_k)
