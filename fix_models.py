# fix_models.py

import os
import torch
import re
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

ROOT_MODEL_DIR = "sae_gridsearch_2"
FIXED_MODEL_DIR = "sae_gridsearch_2_fixed"

os.makedirs(FIXED_MODEL_DIR, exist_ok=True)

def parse_model_filename(filename):
    arch = filename.split("_")[0]
    exp = int(re.search(r"exp(\d+)", filename).group(1))
    lr = float(re.search(r"lr([\d.]+)", filename).group(1))
    l1 = float(re.search(r"l1([\d.]+)", filename).group(1))
    topk = int(re.search(r"topk(\d+)", filename).group(1)) if "topk" in filename else None
    bw = float(re.search(r"bw([\d.]+)", filename).group(1)) if "bw" in filename else None
    return arch, exp, lr, l1, topk, bw

for rat_dir in os.listdir(ROOT_MODEL_DIR):
    rat_path = os.path.join(ROOT_MODEL_DIR, rat_dir)
    if not os.path.isdir(rat_path):
        continue

    fixed_rat_path = os.path.join(FIXED_MODEL_DIR, rat_dir)
    os.makedirs(fixed_rat_path, exist_ok=True)

    emb_dim = int(rat_dir.split("_")[1])  # gatsby_64 → 64

    for fname in os.listdir(rat_path):
        if not fname.endswith(".pt"):
            continue

        model_path = os.path.join(rat_path, fname)
        fixed_model_path = os.path.join(fixed_rat_path, fname)

        try:
            arch, exp, lr, l1, topk, bw = parse_model_filename(fname)
            model_obj = torch.load(model_path, map_location="cpu")
            model = get_model(arch, emb_dim, exp, l1=l1, topk=topk, bw=bw)
            model.load_state_dict(model_obj.state_dict() if hasattr(model_obj, 'state_dict') else model_obj)
            torch.save(model.state_dict(), fixed_model_path)
            print(f"Saved: {fixed_model_path}")
        except Exception as e:
            print(f"Failed on {model_path}: {e}")
