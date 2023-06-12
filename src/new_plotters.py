import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .tools import ewma, freeze

import torch
import gc


def plot_fixed_sde_images(X, Y, T, n_samples=4, gray=False):
    assert len(X) == 10 and len(Y) == 10
    with torch.no_grad():
        T_X = torch.stack([T(X)[0] for i in range(n_samples)], dim=0)
    
    imgs = torch.cat((X[None, :], T_X, Y[None, :]), dim=0).detach().cpu().permute(0, 1, 3, 4, 2)
    imgs = imgs.mul(0.5).add(0.5).numpy().clip(0,1)
    
    fig, axes = plt.subplots(2+n_samples, 10, figsize=(15, 9), dpi=150)
    for i in range(2+n_samples):
        for j in range(10):
            if not gray:
                axes[i][j].imshow(imgs[i][j])
            else:
                axes[i][j].imshow(imgs[i][j], cmap='gray', vmin=0, vmax=1)
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].set_yticks([])
            
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(n_samples):
        axes[i+1, 0].set_ylabel('T(X)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    return fig, axes


def plot_random_sde_images(X_sampler, Y_sampler, T, n_samples=4, gray=False):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)
    
    return plot_fixed_sde_images(X, Y, T, n_samples, gray)


def plot_several_fixed_sde_trajectories(X, Y, T, steps_to_show, times, gray=False):    
    n_trajectories=3
    X = torch.repeat_interleave(X[:4], repeats=n_trajectories, dim=0)
    Y = torch.repeat_interleave(Y[:4], repeats=n_trajectories, dim=0)
    
    with torch.no_grad():
        trajectory = T(X, return_trajectory=True)[0]
        trajectory = torch.transpose(trajectory, 0, 1)
        trajectory = torch.stack([trajectory[step] for step in steps_to_show], dim=0)
        
    imgs = torch.cat((X[None, :], trajectory, Y[None,  :]), dim=0).detach().cpu().permute(0, 1, 3, 4, 2)
    imgs = imgs.mul(0.5).add(0.5).numpy().clip(0,1)
    
    fig, axes = plt.subplots(2+len(steps_to_show), 12, figsize=(15, 20), dpi=150)
    for i in range(2+len(steps_to_show)):
        for j in range(12):
            if not gray:
                axes[i][j].imshow(imgs[i][j])
            else:
                axes[i][j].imshow(imgs[i][j], cmap='gray', vmin=0, vmax=1)
            axes[i][j].get_xaxis().set_visible(False)
            axes[i][j].set_yticks([])
    
    axes[0, 0].set_ylabel('X', fontsize=16)
    for i, step in enumerate(steps_to_show):
        axes[i+1, 0].set_ylabel(f'T(X)_{round(times[step], 4)}', fontsize=16)
#     axes[-2, 0].set_ylabel('T(X)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=16)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes


def plot_several_random_sde_trajectories(X_sampler, Y_sampler, T, steps_to_show, times, gray=False):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)
    
    return plot_several_fixed_sde_trajectories(X, Y, T, steps_to_show, times, gray)
