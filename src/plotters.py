import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .tools import ewma, freeze

import torch
import gc


def plot_fixed_sde_images(X, Y, T, n_samples=4, gray=False):
    freeze(T);
    with torch.no_grad():
        T_X = torch.stack([T(X)[0][:, -1] for i in range(n_samples)], dim=0)
        c, h, w = X.shape[1:]
        imgs = torch.cat([X[None, :], T_X, Y[None, :]]).reshape(-1, c, h, w).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(2+n_samples, 10, figsize=(15, 9), dpi=150)
    
    for i, ax in enumerate(axes.flatten()):
        if not gray:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(n_samples):
        axes[i+1, 0].set_ylabel('T(X)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes


def plot_random_sde_images(X_sampler, Y_sampler, T, n_samples=4, gray=False):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)
    
    return plot_fixed_sde_images(X, Y, T, n_samples, gray)


def plot_fixed_sde_trajectories(X, Y, T, n_steps_to_show=10, n_steps=10, gray=False):
    freeze(T);
    
    trajectory_index_step_size = n_steps // n_steps_to_show
    with torch.no_grad():
        trajectory = T(X)[0]
        T_X = torch.cat((trajectory[:, ::trajectory_index_step_size], trajectory[:, -1][:, None]), dim=1)
        T_X = torch.transpose(T_X, 0, 1)[1:]
        
        c, h, w = X.shape[1:]
        imgs = torch.cat([X[None, :], T_X, Y[None, :]]).reshape(-1, c, h, w).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)
    
    rows_with_steps = np.arange(n_steps + 1)[::trajectory_index_step_size].shape[0]
    fig, axes = plt.subplots(2+rows_with_steps, 10, figsize=(15, 20), dpi=150)
    
    for i, ax in enumerate(axes.flatten()):
        if not gray:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(rows_with_steps):
        axes[i+1, 0].set_ylabel(f'T(X)_{(i+1)*trajectory_index_step_size}', fontsize=24)
    axes[-2, 0].set_ylabel('T(X)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes


def plot_fixed_sde_trajectories(X, Y, T, n_steps_to_show=10, n_steps=10, gray=False):
    freeze(T);
    
    trajectory_index_step_size = n_steps // n_steps_to_show
    with torch.no_grad():
        trajectory = T(X)[0]
        T_X = torch.cat((trajectory[:, ::trajectory_index_step_size], trajectory[:, -1][:, None]), dim=1)
        T_X = torch.transpose(T_X, 0, 1)[1:]
        
        c, h, w = X.shape[1:]
        imgs = torch.cat([X[None, :], T_X, Y[None, :]]).reshape(-1, c, h, w).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)
    
    rows_with_steps = np.arange(n_steps + 1)[::trajectory_index_step_size].shape[0]
    fig, axes = plt.subplots(2+rows_with_steps, 10, figsize=(15, 20), dpi=150)
    
    for i, ax in enumerate(axes.flatten()):
        if not gray:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(rows_with_steps):
        axes[i+1, 0].set_ylabel(f'T(X)_{(i+1)*trajectory_index_step_size}', fontsize=24)
    axes[-2, 0].set_ylabel('T(X)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes


def plot_random_sde_trajectories(X_sampler, Y_sampler, T, n_steps_to_show=10, n_steps=10, gray=False):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)
    
    return plot_fixed_sde_trajectories(X, Y, T, n_steps_to_show, n_steps, gray)


def plot_several_fixed_sde_trajectories(X, Y, T, n_steps_to_show=10, n_steps=10, gray=False):
    freeze(T);
    
    n_trajectories=3
    X = X[:4]
    Y = Y[:4]
    
    trajectory_index_step_size = n_steps // n_steps_to_show
    with torch.no_grad():
        trajectory = []
        for i in range(n_trajectories):
            trajectory.append(T(X)[0])
        
        trajectory = torch.stack(trajectory, dim=1)
        trajectory = trajectory.reshape(trajectory.shape[0]*trajectory.shape[1], trajectory.shape[2], trajectory.shape[3], trajectory.shape[4], trajectory.shape[5])
        
        T_X = torch.cat((trajectory[:, ::trajectory_index_step_size], trajectory[:, -1][:, None]), dim=1)
        T_X = torch.transpose(T_X, 0, 1)[1:]
        
        c, h, w = X.shape[1:]
        imgs = torch.cat([X.repeat_interleave(3, dim=0)[None, :], T_X, Y.repeat_interleave(3, dim=0)[None, :]]).reshape(-1, c, h, w).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)
    
    rows_with_steps = np.arange(n_steps + 1)[::trajectory_index_step_size].shape[0]
    fig, axes = plt.subplots(2+rows_with_steps, 12, figsize=(15, 20), dpi=150)
    
    for i, ax in enumerate(axes.flatten()):
        if not gray:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(rows_with_steps):
        axes[i+1, 0].set_ylabel(f'T(X)_{(i+1)*trajectory_index_step_size}', fontsize=24)
    axes[-2, 0].set_ylabel('T(X)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes


def plot_several_random_sde_trajectories(X_sampler, Y_sampler, T, n_steps_to_show=10, n_steps=10, gray=False):
    X, Y = X_sampler.sample(10), Y_sampler.sample(10)
    
    return plot_several_fixed_sde_trajectories(X, Y, T, n_steps_to_show, n_steps, gray)
    

def plot_Z_images(XZ, Y, T):
    freeze(T);
    with torch.no_grad():
        T_XZ = T(
            *(XZ[0].flatten(start_dim=0, end_dim=1), XZ[1].flatten(start_dim=0, end_dim=1))
        ).permute(1,2,3,0).reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4).permute(4,3,0,1,2).flatten(start_dim=0, end_dim=1)
        imgs = torch.cat([XZ[0][:,0], T_XZ, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(6, 10, figsize=(15, 9), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(4):
        axes[i+1, 0].set_ylabel('T(X,Z)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_Z_images(X_sampler, ZC, Z_STD, Y_sampler, T):
    X = X_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z = torch.randn(10, 4, ZC, 1, 1, device='cuda') * Z_STD
        XZ = (X, Z,)
    Y = Y_sampler.sample(10)
    return plot_Z_images(XZ, Y, T)


def plot_bar_and_stochastic_2D(X_sampler, Y_sampler, T, ZD, Z_STD):
    DIM = 2
    freeze(T)
    
    PLOT_X_SIZE_LEFT = 64
    PLOT_Z_COMPUTE_LEFT = 256

    PLOT_X_SIZE_RIGHT = 32
    PLOT_Z_SIZE_RIGHT = 4

    assert PLOT_Z_COMPUTE_LEFT >= PLOT_Z_SIZE_RIGHT
    assert PLOT_X_SIZE_LEFT >= PLOT_X_SIZE_RIGHT

    X = X_sampler.sample(PLOT_X_SIZE_LEFT).reshape(-1, 1, DIM).repeat(1, PLOT_Z_COMPUTE_LEFT, 1)
    Y = Y_sampler.sample(PLOT_X_SIZE_LEFT)

    with torch.no_grad():
        Z = torch.randn(PLOT_X_SIZE_LEFT, PLOT_Z_COMPUTE_LEFT, ZD, device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1, 0).reshape(DIM, -1, PLOT_Z_COMPUTE_LEFT).permute(1, 2, 0)

    X_np = X[:, 0].cpu().numpy()
    XZ_np = T_XZ.cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_XZ_np = T_XZ.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=150, sharex=True, sharey=True, )
    for i in range(2):
        axes[i].set_xlim(-2.5, 2.5); axes[i].set_ylim(-2.5, 2.5)
        axes[i].grid(True)

#     axes[0].set_title(r'Map $x\mapsto \overline{T}(x)=\int_{\mathcal{Z}}T(x,z)d\mathbb{S}(z)$', fontsize=22, pad=10)
#     axes[1].set_title(r'Stochastic map $x\mapsto T(x,z)$', fontsize=20, pad=10)

    from matplotlib import collections  as mc
    lines = list(zip(X_np[:PLOT_X_SIZE_LEFT], T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT]))

    lc = mc.LineCollection(lines, linewidths=1, color='black')
    axes[0].add_collection(lc)

    axes[0].scatter(
        X_np[:PLOT_X_SIZE_LEFT, 0], X_np[:PLOT_X_SIZE_LEFT, 1], c='lightcoral', edgecolors='black',
        zorder=2, label=r'$x\sim\mathbb{P}$'
    )
    axes[0].scatter(
        T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT, 0], T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT, 1],
        c='wheat', edgecolors='black', zorder=2, label=r'$\overline{T}(x)$', marker='v'
    )
    axes[0].legend(fontsize=12, loc='lower right', framealpha=1)

    lines = []
    for i in range(PLOT_X_SIZE_RIGHT):
        for j in range(PLOT_Z_SIZE_RIGHT):
            lines.append((X_np[i], T_XZ_np[i, j]))
    lc = mc.LineCollection(lines, linewidths=0.5, color='black')
    axes[1].add_collection(lc)
    axes[1].scatter(
        X_np[:PLOT_X_SIZE_RIGHT, 0], X_np[:PLOT_X_SIZE_RIGHT, 1], c='lightcoral', edgecolors='black',
        zorder=2,  label=r'$x\sim\mathbb{P}$'
    )
    axes[1].scatter(
        T_XZ_np[:PLOT_X_SIZE_RIGHT, :PLOT_Z_SIZE_RIGHT, 0].flatten(),
        T_XZ_np[:PLOT_X_SIZE_RIGHT, :PLOT_Z_SIZE_RIGHT, 1].flatten(),
        c='darkseagreen', edgecolors='black', zorder=3,  label=r'$\widehat{T}(x,z)$'
    )
    axes[1].legend(fontsize=12, loc='lower right', framealpha=1)

    fig.tight_layout()
    
    return fig, axes

def plot_generated_2D(X_sampler, Y_sampler, T, ZD, Z_STD):
    DIM = 2
    freeze(T)

    PLOT_SIZE = 512
    X = X_sampler.sample(PLOT_SIZE).reshape(-1, 1, DIM).repeat(1, 1, 1)
    Y = Y_sampler.sample(PLOT_SIZE)

    with torch.no_grad():
        Z = torch.randn(PLOT_SIZE, 1, ZD, device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1, 0).reshape(DIM, -1, 1).permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True, dpi=150)

    X_np = X[:,0].cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_XZ_np = T_XZ[:,0].cpu().numpy()

    for i in range(3):
        axes[i].set_xlim(-2.5, 2.5); axes[i].set_ylim(-2.5, 2.5)
        axes[i].grid(True)

    axes[0].scatter(X_np[:, 0], X_np[:, 1], c='lightcoral', edgecolors='black', label=r'Input $x\sim\mathbb{P}$',zorder=2,)
    axes[1].scatter(Y_np[:, 0], Y_np[:, 1], c='darkseagreen', edgecolors='black', label=r'Target $y\sim\mathbb{Q}$',zorder=2,)
    axes[2].scatter(T_XZ_np[:, 0], T_XZ_np[:, 1], c='darkseagreen', edgecolors='black', label=r'Mapped $\widehat{T}(x,z)$',zorder=2,)
    for i in range(3):
        axes[i].legend(fontsize=12, loc='lower right', framealpha=1)
    
#     axes[0].set_title(r'Input $x\sim\mathbb{P}$', fontsize=22, pad=10)
#     axes[1].set_title(r'Target $y\sim\mathbb{Q}$', fontsize=22, pad=10)
#     axes[2].set_title(r'Fitted $T(x,z)_{\#}(\mathbb{P}\times\mathbb{S})$', fontsize=22, pad=10)

    fig.tight_layout()
    return fig, axes