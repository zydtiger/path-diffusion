# Path Diffusion

Train a basic DDPM on PathMNIST.

## Current Model

The current implementation is a standard DDPM:

- Backbone: U-Net noise predictor
- Objective: predict added Gaussian noise with MSE loss
- Forward process: linear beta schedule over 1000 diffusion steps
- Dataset: PathMNIST (`28x28`, `3` channels)

## Setup

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -e .
```

## Run Training

```bash
python main.py
```

Use DiT instead of U-Net:

```bash
python main.py --dit
```

Use MedMNIST+ resolution (`224x224`):

```bash
python main.py --large
```

This will:

- train the model
- log batch loss to TensorBoard in `runs/`
- save generated samples to `outputs/generated_samples.png`
- save model weights to `diffusion_model.pt`

## TensorBoard

```bash
tensorboard --logdir runs
```
