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

## Repository structure
The implementation is GPU-based with the multi-GPU support.

All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). For convenience, the majority of the evaluation output is preserved. Auxilary source code is moved to `.py` modules (`src/`). 

Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

- ```notebooks/Toy_experiments.ipynb``` - Toy experiments.
- ```notebooks/High_dimensionsal_gaussians.ipynb``` - Experiments with high dimensional gaussians.
- ```stats/compute_stats.ipynb``` - Precomputing stats for FID evalution for colored MNIST and Celeba (you need to run it before experiments with images).
- ```notebooks/Image_experiments.ipynb``` - Training ENOT for colored MNIST and Celeba.
- ```notebooks/Discrete_OT.ipynb``` - Calculating discrete OT mappings.
- ```notebooks/MNIST_plotting.ipynb``` - Plotting ENOT and discrete OT results for colored MNIST.
- ```notebooks/Celeba_plotting.ipynb``` - Plotting ENOT results for Celeba.

## Datasets
- Colored MNIST. Custom dataseted obtained by coloring each MNIST digith in a random color;
- [CelebA faces](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) requires ```datasets/list_attr_celeba.ipynb```;

The dataloaders can be created by ```load_dataset``` function from ```src/tools.py```. The latter four datasets get loaded directly to RAM.
