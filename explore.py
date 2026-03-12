import os
import torch
from torchvision.utils import save_image
from medmnist import PathMNIST
from torchvision import transforms


def main():
    # Create transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1),
        ]
    )

    # Load train dataset
    train_dataset = PathMNIST(split="train", download=True)
    train_dataset.transform = transform

    # Ensure samples directory exists
    os.makedirs("samples", exist_ok=True)

    # Get first 8 images and save them
    images = []
    for i in range(8):
        img, _ = train_dataset[i]
        images.append(img)

    # Stack images into a batch
    images = torch.stack(images)

    # Save images
    save_image(images, "samples/first_8_images.png", nrow=4, normalize=True)
    print("Saved first 8 images to samples/first_8_images.png")


if __name__ == "__main__":
    main()