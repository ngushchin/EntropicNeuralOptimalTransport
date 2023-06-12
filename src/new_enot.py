import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl

from .resnet2 import ResNet_D
from .cunet import CUNet


def get_euler_maruyama_solution(sde, x0, times, 
                                create_graph=False,
                                return_trajectory=False,
                                n_last_steps_without_noise=0):
    delta_times = times[1:] - times[:-1]
    
    trajectory = [x0]
    
    for i, (t, delta_t) in enumerate(zip(times[:-1], delta_times)):
        drifts = sde.get_drift(x0, t, delta_t=delta_t)
        epsilons = sde.get_var(x0, t)
        
        if i >= len(times) - 1 - n_last_steps_without_noise:
            epsilons = [elem*0 for elem in epsilons]
        
        x0 = tuple([
            x_elem + drift*delta_t + torch.sqrt(epsilon*delta_t)*torch.randn_like(x_elem, device=x_elem.device) for x_elem, drift, epsilon in zip(x0, drifts, epsilons)
        ])
        if not create_graph:
            x0 = tuple(elem.detach() for elem in x0)
            
        if return_trajectory:
            trajectory.append(x0)
    
    if return_trajectory:
        trajectory = list(zip(*trajectory))
        return tuple([torch.stack(elem, dim=1) for elem in trajectory])
    
    return x0


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.silu(input)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()

        self.dim = dim
        self.scale = scale

        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000)) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        
        input = input*self.scale + 1
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb
    

class DriftModel(nn.Module):
    def __init__(self, dataset_1_channels, dataset_2_channels, 
                 unet_base_factor, n_steps, time_dim, 
                  predict_drift_from_new_state=True):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim, scale=n_steps),
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
        self.model = CUNet(dataset_1_channels, dataset_2_channels,
                  time_dim, base_factor=unet_base_factor)
        self.predict_drift_from_new_state = predict_drift_from_new_state
    
    def forward(self, x, t, **kwargs):        
        t = t.expand(x.shape[0])
        t = self.time_embed(t)
        t = t[:, :, None, None]
        
        if self.predict_drift_from_new_state:
            assert "delta_t" in kwargs
            drifted_x = self.model(x, t)
            x_drift = (drifted_x - x)/kwargs["delta_t"]
        else:
            x_drift = self.model(x, t)
        
        return x_drift


class EnotSDE(nn.Module):
    def __init__(self, dataset_1_channels, dataset_2_channels, 
                 unet_base_factor, epsilon,
                 times, n_last_steps_without_noise, time_dim,
                 predict_drift_from_new_state=True):
        super().__init__()
        self.register_buffer("times", torch.tensor(times), persistent=False)
        n_steps = len(times) - 1
        self.drift_model = DriftModel(dataset_1_channels=dataset_1_channels,
                                      dataset_2_channels=dataset_2_channels,
                                      time_dim=time_dim, unet_base_factor=unet_base_factor,
                                      n_steps=n_steps, predict_drift_from_new_state=predict_drift_from_new_state)
        self.n_last_steps_without_noise=n_last_steps_without_noise
        self.register_buffer("epsilon", torch.tensor(epsilon), persistent=False)
    
    def forward(self, x, create_graph=False, return_trajectory=False):
        norm_square = torch.zeros(x.shape[0], device=x.device)
        x0 = (x, norm_square)
                     
        x, norm_square = get_euler_maruyama_solution(self, x0, self.times,
                                                     create_graph=create_graph,
                                                     return_trajectory=return_trajectory,
                                                     n_last_steps_without_noise=self.n_last_steps_without_noise)
        return x, norm_square
    
    def get_drift(self, x, t, **kwargs):
        x_coordinate, drift_norm_square = x
            
        x_coordinate_drift = self.drift_model(x_coordinate, t, **kwargs)
        x_norm_square_drift = torch.norm(torch.flatten(x_coordinate_drift, start_dim=1, end_dim=- 1), p=2, dim=-1).square()
        
        return (x_coordinate_drift, x_norm_square_drift)
    
    def get_var(self, x, t):
        return (self.epsilon, 0)

    def set_epsilon(self, epsilon):
        self.register_buffer("epsilon", torch.tensor(epsilon, device=next(self.parameters()).device), persistent=False)

        
class LitENOT(pl.LightningModule):        
    def __init__(
        self,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.D = ResNet_D(self.hparams.img_size, nc=self.hparams.dataset_2_channels)
        self.D.apply(weights_init_D)
        
        self.T = EnotSDE(dataset_1_channels=self.hparams.dataset_1_channels,
                                dataset_2_channels=self.hparams.dataset_2_channels, 
                                unet_base_factor=self.hparams.unet_base_factor,
                                epsilon=self.hparams.epsilon,
                                times=self.hparams.times, time_dim=self.hparams.time_dim,
                                predict_drift_from_new_state=self.hparams.predict_drift_from_new_state)
        
        epsilon = self.hparams.epsilon
        epsilon_scheduler_last_iter = self.hparams.epsilon_scheduler_last_iter
        self.epsilon_scheduler = (
            lambda step: min(epsilon, epsilon*(step/(epsilon_scheduler_last_iter)))
        )
        self.inner_iters = self.hparams.inner_iters
        self.norm_square_normalization_constant = self.hparams.norm_square_normalization_constant
        self.fid_metric = FrechetInceptionDistance(feature=2048, reset_real_features=True, normalize=True)
        

    def training_step(self, batch):
        outer_optimization_step = self.global_step // (self.inner_iters + 1)
        is_generetor_optimization_step = self.global_step % (self.inner_iters + 1) != 0
        
        T_opt, D_opt = self.optimizers()
        opt = T_opt if is_generetor_optimization_step else D_opt
        
        new_epsilon = self.epsilon_scheduler(outer_optimization_step)
        self.T.set_epsilon(new_epsilon)
        
        (X, _), (X1, _) = batch
        XN, norm_square = self.T(X, create_graph=True)
        norm_square = self.norm_square_normalization_constant*norm_square
        
        self.toggle_optimizer(opt)
        opt.zero_grad(set_to_none=True)
        loss = (
            (norm_square - self.D(XN)).mean() if is_generetor_optimization_step
            else (self.D(XN) - self.D(X1)).mean()
        )
        self.manual_backward(loss)
        opt.step()
        self.untoggle_optimizer(opt)
        
        if outer_optimization_step % 10 == 0 and not is_generetor_optimization_step:
#             self.log('loss', loss.item(), step=outer_optimization_step)
            step = outer_optimization_step
            wandb.log({f'Epsilon' : new_epsilon}, step=step)
            wandb.log({f'Mean norm' : torch.sqrt(norm_square).mean().item()}, step=step)
            wandb.log({f'D_loss' : loss.item()}, step=step)
#             wandb.log({f'T_loss' : T_loss.item()}, step=step)

#             wandb.log({f'D_loss' : D_loss.item()}, step=step)
            wandb.log({f'D_X1' : self.D(X1).mean().item()}, step=step)
            wandb.log({f'D_XN' : self.D(XN).mean().item()}, step=step)

        
    def configure_optimizers(self):
        T_lr = self.hparams.T_lr
        D_lr = self.hparams.D_lr

        opt_D = torch.optim.Adam(self.D.parameters(), lr=D_lr, weight_decay=1e-10)
        opt_T = torch.optim.Adam(self.T.parameters(), lr=T_lr, weight_decay=1e-10)
        return [opt_T, opt_D], []
    
    

        