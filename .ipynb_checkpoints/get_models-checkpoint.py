import sys
import numpy as np
import matplotlib.pyplot as plt
import cebra
from cebra import CEBRA
from sklearn.model_selection import train_test_split
import os
import tempfile
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
import pandas as pd

rats = ['achilles', 'buddy', 'cicero', 'gatsby']

datasets = {}

train_data = {}

for name in rats:
    datasets[name] = cebra.datasets.hippocampus.SingleRatDataset(name=name, root='data', download=True)

    split_idx = int(0.8 * len(datasets[name].neural))
    train_data[name] = datasets[name].neural[:split_idx]
    
    
params_grid = dict(
    output_dimension=[8, 16, 32, 64],
    model_architecture=['offset10-model'],
    time_offsets=[5],
    temperature_mode='constant',
    temperature=[0.1],
    max_iterations=[2000],
    num_hidden_units=[64],
    device='cuda_if_available',
    distance = ["cosine"],
    batch_size = 256,
    learning_rate = [1e-3],
    verbose=True
)
grid_search = cebra.grid_search.GridSearch()
models, params = grid_search.generate_models(params=params_grid)
    
grid_search.fit_models(train_data, params=params_grid, models_dir="embedding_models")