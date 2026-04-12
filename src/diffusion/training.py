import copy

import torch
import torch.nn.functional as F
from prettyterm import track
from torch.utils.tensorboard import SummaryWriter

from diffusion.scheduler import LinearNoiseScheduler


@torch.no_grad()
def _ema_update(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def train_epoch(
    epoch: int,
    model: torch.nn.Module,
    dataloader,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    device: str,
    writer: SummaryWriter,
    num_batches: int,
    ema_model: torch.nn.Module | None = None,
    ema_decay: float = 0.999,
):
    model.train()

    pbar = track(dataloader, desc=f"Epoch {epoch + 1}")
    for i, batch in enumerate(pbar):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        noisy_images, t, noise = scheduler.q_sample_batched(images)
        predicted_noise = model(noisy_images, t)

        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema_model is not None:
            _ema_update(ema_model, model, ema_decay)

        if i % 10 == 0:
            global_step = epoch * num_batches + i
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            pbar.set_postfix({"loss": f"{loss:.3f}"})


def train(
    model: torch.nn.Module,
    train_loader,
    scheduler: LinearNoiseScheduler,
    device: str,
    num_epochs: int = 50,
    lr: float = 1e-4,
    ema_decay: float = 0.999,
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    writer = SummaryWriter(log_dir="runs/")
    num_batches = len(train_loader)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)

    try:
        for epoch in range(num_epochs):
            train_epoch(
                epoch,
                model,
                train_loader,
                scheduler,
                optimizer,
                device,
                writer,
                num_batches,
                ema_model=ema_model,
                ema_decay=ema_decay,
            )
            scheduler_opt.step()
    finally:
        writer.close()

    model.load_state_dict(ema_model.state_dict())
    return model
