# Entropic Neural Optimal Transport via Diffusion Processes

This repository contains code to reproduce the experiments from our work [https://arxiv.org/abs/2211.01156](https://arxiv.org/abs/2211.01156). PyTorch implementation.

## Citation

If you find this repository or the ideas presented in our paper useful, please consider citing our paper.

```
@article{gushchin2022entropic,
  title={Entropic Neural Optimal Transport via Diffusion Processes},
  author={Gushchin, Nikita and Kolesov, Alexander and Korotin, Alexander and Vetrov, Dmitry and Burnaev, Evgeny},
  journal={arXiv preprint arXiv:2211.01156},
  year={2022}
}
```

## Experiments 

Below, we give the instructions how to launch the experiments from our manuscript. Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

### Toy 2D experiment (2D Gaussian to Swissroll and Gaussian to 8 Gaussians)

```notebooks/Toy_experiments.ipynb``` - Toy experiments.

### High-dimensional Gaussians

```notebooks/High_dimensionsal_gaussians.ipynb``` - Experiments with high dimensional gaussians.

###  CelebA unpaired debluring and ColoredMnist 2 to 3.

```stats/compute_stats.ipynb``` - Precomputing stats for FID evalution for colored MNIST and Celeba (you need to run it before experiments with images).

```notebooks/Image_experiments.ipynb``` - Training ENOT for colored MNIST and Celeba.

```notebooks/Discrete_OT.ipynb``` - Calculating discrete OT mappings.

```notebooks/MNIST_plotting.ipynb``` - Plotting ENOT and discrete OT results for colored MNIST.

```notebooks/Celeba_plotting.ipynb``` - Plotting ENOT results for Celeba.
