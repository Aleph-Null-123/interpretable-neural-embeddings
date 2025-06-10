import os
import numpy as np
from pathlib import Path
import cebra

import torch


rats = ['achilles', 'buddy', 'cicero', 'gatsby']

datasets = {}

train_data = {}
valid_data = {}

for name in rats:
    datasets[name] = cebra.datasets.hippocampus.SingleRatDataset(name=name, root='data', download=True)

    split_idx = int(0.8 * len(datasets[name].neural))
    train_data[name] = datasets[name].neural[:split_idx]
    valid_data[name] = datasets[name].neural[split_idx:]
    

MODEL_DIR = "final_models"
EMBEDDING_SAVE_DIR = "final_embeddings"
os.makedirs(EMBEDDING_SAVE_DIR, exist_ok=True)

models, parameter_grid = cebra.grid_search.GridSearch().load(dir="embedding_models")

for model_name in models:
    print(f"Loading model: {model_name}")
    
    model = models[model_name]
    
    rat = model_name.split("_")[-1]

    train_embedding = model.transform(train_data[rat])
    valid_embedding = model.transform(valid_data[rat])
    
    
    outdir = f"{EMBEDDING_SAVE_DIR}/{model_name}"
    os.makedirs(outdir, exist_ok=True)
    
    
    print(train_embedding, valid_embedding)

    np.save(f"{outdir}/train_embeddings.npy", train_embedding)
    np.save(f"{outdir}/valid_embeddings.npy", valid_embedding)

    print(f"Saved embeddings to {outdir}")
