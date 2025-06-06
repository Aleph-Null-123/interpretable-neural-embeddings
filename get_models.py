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

rats = ['achilles', 'buddy', 'cicero', 'gatsby']

datasets = {}

train_data = {}

valid_data = {}

train_label = {}

valid_label = {}

for name in rats:
    datasets[name] = cebra.datasets.hippocampus.SingleRatDataset(name=name, root='data', download=True)


    split_idx = int(0.8 * len(datasets[name].neural))
    train_data[name] = datasets[name].neural[:split_idx]
    valid_data[name] = datasets[name].neural[split_idx:]

    train_label[name] = datasets[name].continuous_index.numpy()[:split_idx]
    valid_label[name] = datasets[name].continuous_index.numpy()[split_idx:]
    
    
params_grid = dict(
    output_dimension=[8, 16, 32, 64],
    model_architecture=['offset10-model'],
    time_offsets=[5, 10],
    temperature_mode='constant',
    temperature=[0.1, 0.5, 1],
    max_iterations=[1000, 2000, 5000],
    num_hidden_units=[64, 128],
    device='cuda_if_available',
    distance = ["cosine"],
    learning_rate = [1e-4, 3e-4, 5e-4, 1e-3],
    verbose=True
)

grid_search = cebra.grid_search.GridSearch()
grid_search.fit_models(train_data, params=params_grid, models_dir="saved_models")

df_results = grid_search.get_df_results(models_dir="saved_models")
t = datetime.now()
df_results.to_csv(f'gridsearch_results/{t}.csv', index = False)

for name in rats:
    best_model, best_model_name = grid_search.get_best_model(dataset_name=name, models_dir="saved_models")
    with open(f"gridsearch_results/{t}.log", "a") as f:
          f.write(f"The best model for datset {name} is {best_model_name}")
    print(f"The best model for datset {name} is {best_model_name}")


    
    
