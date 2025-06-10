import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_position_r2(latents, position):
    model = LinearRegression()
    model.fit(latents, position)
    pred = model.predict(latents)
    return r2_score(position, pred)


def compute_direction_acc(latents, direction):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(latents, direction)
    pred = clf.predict(latents)
    return accuracy_score(direction, pred)


def compute_behavior_corr(latents, behavior_labels):
    return np.corrcoef(latents.T, behavior_labels.T)[:latents.shape[1], latents.shape[1]:]


def compute_sparsity(latents, threshold=1e-3):
    active = (np.abs(latents) > threshold).sum(axis=1)
    return np.mean(active / latents.shape[1]) * 100


def neuron_latent_attribution(neural_data, latents):
    return np.corrcoef(neural_data.T, latents.T)[:neural_data.shape[1], neural_data.shape[1]:]
