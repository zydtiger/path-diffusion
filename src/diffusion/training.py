import torch
import torch.nn.functional as F
from prettyterm import track
from torch.utils.tensorboard import SummaryWriter

from diffusion.scheduler import LinearNoiseScheduler


def train_epoch(
    epoch: int,
    model: torch.nn.Module,
    dataloader,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    device: str,
    writer: SummaryWriter,
    num_batches: int,
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
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler_opt = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    writer = SummaryWriter(log_dir="runs/")
    num_batches = len(train_loader)

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
            )
            scheduler_opt.step()
    finally:
        writer.close()

    return model
