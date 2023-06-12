import torch
import torch.nn as nn
import math
import numpy as np
import pdb
import torch.nn.functional as F

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
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000) / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        
        input = input*self.scale + 1
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class SDE(nn.Module):
    def __init__(self, shift_model, epsilon, n_steps,
                 time_dim, n_last_steps_without_noise,
                 use_positional_encoding, use_gradient_checkpoint,
                 predict_shift, image_input=False):
        super().__init__()
        self.shift_model = shift_model
        self.epsilon = epsilon
        self.n_steps = n_steps
        self.n_last_steps_without_noise = n_last_steps_without_noise
        self.use_positional_encoding = use_positional_encoding
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.times = np.linspace(0, 1, n_steps+1).tolist()
        self.predict_shift = predict_shift
        self.image_input = image_input
        
        self.time = nn.Sequential(
            TimeEmbedding(time_dim, scale=n_steps),
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim),
        )
    
    def forward(self, x0, return_trajectory=True):            
        t0 = 0.0
        trajectory = [x0]
        times = [t0]
        shifts = []

        x, t = x0, t0

        for i, t_next in enumerate(self.times[1:]):

            if i >= len(self.times[1:]) - self.n_last_steps_without_noise:
                x, shift = self._step(x, t, t_next - t, add_noise=False)
            else:
                x, shift = self._step(x, t, t_next - t, add_noise=True)

            t = t_next
            
            if return_trajectory:
                trajectory.append(x)
                times.append(t)
                shifts.append(shift)
                    
        if not return_trajectory:
            trajectory.append(x)
            times.append(t)
            shifts.append(shift)
            
        trajectory = torch.stack(trajectory, dim=1)
        times = torch.tensor(times)[None, :].repeat(trajectory.shape[0], 1).cuda()
        shifts = torch.stack(shifts, dim=1)
        
        return trajectory, times, shifts
    
    def _step(self, x, t, delta_t, add_noise=True):
        if self.predict_shift:
            shift_dt = self._get_shift(x, t)
            shifted_x = x + shift_dt
            shift = shift_dt/(torch.tensor(delta_t).cuda())
        else:
            shifted_x = self._get_shift(x, t)
            shift = (shifted_x - x)/(torch.tensor(delta_t).cuda())
        noise = self._sample_noise(x, delta_t)
        
        if add_noise:
            return shifted_x + noise, shift
        
        return shifted_x, shift
    
    def _get_shift(self, x, t):
        batch_size = x.shape[0]
        
        if self.use_positional_encoding:
            t = torch.tensor(t).repeat(batch_size)
            t = t.cuda()
            t = self.time(t)
            
            if self.image_input:
                t = t[:, :, None, None]
        else:
            t = torch.tensor(t).repeat(batch_size)[:, None]
            if self.image_input:
                t = t[:, None, None, None]
                
            if x.device.type == "cuda":
                t = t.cuda()
        
        if self.use_gradient_checkpoint:
            return torch.utils.checkpoint.checkpoint(self.shift_model, x, t)
        
            if not self.image_input:
                x = torch.cat((x, t), dim=1)
                return torch.utils.checkpoint.checkpoint(self.shift_model, x)
        
        if not self.image_input:
            x = torch.cat((x, t), dim=1)
            return self.shift_model(x)
        
        return self.shift_model(x, t)
        
    def _sample_noise(self, x, delta_t):
        noise = math.sqrt(self.epsilon)*math.sqrt(delta_t)*torch.randn(x.shape)
        
        if x.device.type == "cuda":
            noise = noise.cuda()
        return noise
            
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        
        
def make_net(n_inputs, n_outputs, n_layers=3, n_hiddens=100):
    layers = [nn.Linear(n_inputs, n_hiddens), nn.ReLU()]
    
    for i in range(n_layers - 1):
        layers.extend([nn.Linear(n_hiddens, n_hiddens), nn.ReLU()])
        
    layers.append(nn.Linear(n_hiddens, n_outputs))
    
    return nn.Sequential(*layers)
        
        
def integrate(values, times):
    deltas = times[1:] - times[:-1]
    if values.device.type == "cuda":
        deltas = deltas.cuda()
    return (values*deltas[None, :]).sum(dim = 1)
