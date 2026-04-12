# DPPS Sampling for Masked Reconstruction

This project now uses a DPPS-style (Diffusion Posterior Proximal Sampling) inference loop for the masked reconstruction task.

## Core Idea

The diffusion model remains an unconditional prior, but each reverse step performs proximal candidate selection to prefer samples that are most consistent with the measurement.

## Problem Setup

- Measurement model (inpainting): `y = M \odot x0`
- `M` is a binary mask (`1` known, `0` missing)
- Reverse state at step `t`: `x_t`

For this task, the forward operator is `A(x) = M \odot x`.

## DPPS Step (t -> t-1)

1. Compute DDPM posterior parameters from model prediction:

- `eps_theta = model(x_t, t)`
- `x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)`
- `mu_t = c1 * x0_hat + c2 * x_t`
- `sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)`

where

- `c1 = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)`
- `c2 = sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)`

2. Draw `K_t` candidates:

- `x_{t-1}^i = mu_t + sigma_t * z_i`, `z_i ~ N(0, I)`

3. Select the proximal candidate by data consistency:

- target measurement-domain value: `r_t = c2 * A(x_t) + c1 * y`
- choose

`i* = argmin_i ||A(x_{t-1}^i) - r_t||_2^2`

- set `x_{t-1} = x_{t-1}^{i*}`

At `t = 0`, use `x_0 = mu_0`.

## Adaptive Candidate Count

`K_t` is adapted from SNR:

- `lambda_t = alpha_bar_t / (1 - alpha_bar_t)`
- `K_t = max(floor(K_max * (1 - exp(-lambda_t))), 2)`
- and clamped to `<= K_max`

This uses fewer candidates in noisy early steps and more candidates in later detail-sensitive steps.

## Aligned Initialization

Instead of pure white-noise start, initialization is measurement-aligned:

`x_T = sqrt(alpha_bar_T) * A^T y + sqrt(1 - alpha_bar_T) * eps`

For inpainting, `A^T = A = M`, so this injects known-region information from the start.

## Practical Interpretation

- Standard DDPM sampling: random draw from posterior each step.
- DPPS: draw multiple posterior candidates each step, keep the one best matching the measurement model.
- This improves stability and measurement fidelity without retraining the diffusion backbone.
