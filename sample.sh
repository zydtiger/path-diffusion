python reconstruct.py \
  -ckpt ./diffusion_model_unet_28.pt \
  -o outputs/reconstructed_samples_unet_28.png \
  --blur-kernel-size 5 \
  --blur-sigma 0.8 \
  --dc-eta 0.12 \
  --dc-eta-final 0.001

python reconstruct.py \
  -ckpt ./diffusion_model_unet_224.pt \
  -o outputs/reconstructed_samples_unet_224.png \
  --blur-kernel-size 29 \
  --blur-sigma 2 \
  --dc-eta 0.12 \
  --dc-eta-final 0.001

python reconstruct.py \
  --dit \
  -ckpt ./diffusion_model_dit_28.pt \
  -o outputs/reconstructed_samples_dit_28.png \
  --blur-kernel-size 5 \
  --blur-sigma 0.8 \
  --dc-eta 0.12 \
  --dc-eta-final 0.001

python reconstruct.py \
  --dit \
  -ckpt ./diffusion_model_dit_224.pt \
  -o outputs/reconstructed_samples_dit_224.png \
  --blur-kernel-size 29 \
  --blur-sigma 2 \
  --dc-eta 0.12 \
  --dc-eta-final 0.001
