import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sae_architectures import StandardSAE, TopKSAE, JumpReLUSAE, GatedSAE

def get_model(arch, input_dim, exp, l1=1e-3, topk=None, bw=None):
    if arch == "standard":
        return StandardSAE(input_dim, expansion_factor=exp, sparsity_lambda=l1)
    if arch == "topk":
        return TopKSAE(input_dim, expansion_factor=exp, topk=topk)
    if arch == "jumprelu":
        return JumpReLUSAE(input_dim, expansion_factor=exp, bandwidth=bw, sparsity_lambda=l1)
    if arch == "gated":
        return GatedSAE(input_dim, expansion_factor=exp, sparsity_lambda=l1)

def train_sae(model, train_tensor, lr, epochs, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            recon, latent, sparse_loss = model(batch)
            loss = loss_fn(recon, batch) + sparse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def save_latents(model, data_tensor, path, device=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    loader = DataLoader(data_tensor, batch_size=512)
    latents = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, z, _ = model(batch)
            latents.append(z.cpu().numpy())
    np.save(path, np.concatenate(latents))


def run_all_saes(emb_path, save_dir, architectures, expansion_factors,
    l1_vals, lr_vals, topk_vals=[None], bw_vals=[None], epochs=50):
    os.makedirs(save_dir, exist_ok=True)
    emb = np.load(emb_path)
    input_dim = emb.shape[1]
    data_tensor = torch.tensor(emb).float()

    for arch in architectures:
        for exp in expansion_factors:
            for lr in lr_vals:
                for l1 in l1_vals:
                    for topk in topk_vals if arch == "topk" else [None]:
                        for bw in bw_vals if arch == "jumprelu" else [None]:
                            name = f"{arch}_exp{exp}_lr{lr}_l1{l1}"
                            if topk: name += f"_topk{topk}"
                            if bw: name += f"_bw{bw}"
                            print("Training:", name)

                            model = get_model(arch, input_dim, exp, l1=l1, topk=topk, bw=bw)
                            model = train_sae(model, data_tensor, lr=lr, epochs=epochs, batch_size = 512)
                            latents_path = os.path.join(save_dir, f"{name}.npy")
                            save_latents(model, data_tensor, latents_path)

                            model_path = os.path.join(save_dir, f"{name}_model.pt")
                            torch.save(model.state_dict(), model_path)
