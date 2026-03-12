import torch
import torch.nn.functional as F

from diffusion.scheduler import LinearNoiseScheduler


def train_epoch(
    model: torch.nn.Module,
    dataloader,
    scheduler: LinearNoiseScheduler,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)

        noisy_images, t, noise = scheduler.q_sample_batched(images)
        predicted_noise = model(noisy_images, t)

        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


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

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, scheduler, optimizer, device)
        scheduler_opt.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    return model
