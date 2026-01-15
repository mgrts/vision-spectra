"""
Image transforms for training and evaluation.
"""

from __future__ import annotations

from torchvision import transforms


def get_train_transforms(image_size: int, num_channels: int) -> transforms.Compose:
    """
    Get training transforms with data augmentation.

    Args:
        image_size: Target image size
        num_channels: Number of image channels (1 or 3)

    Returns:
        Composed transforms
    """
    transform_list = []

    # Resize if needed
    transform_list.append(transforms.Resize((image_size, image_size)))

    # Data augmentation
    transform_list.extend(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1 if num_channels == 3 else 0,
            ),
        ]
    )

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalize
    if num_channels == 1:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(transform_list)


def get_eval_transforms(image_size: int, num_channels: int) -> transforms.Compose:
    """
    Get evaluation transforms (no augmentation).

    Args:
        image_size: Target image size
        num_channels: Number of image channels (1 or 3)

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    # Normalize
    if num_channels == 1:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(transform_list)


def get_mim_transforms(image_size: int, num_channels: int) -> transforms.Compose:
    """
    Get transforms for MIM pretraining.

    Uses lighter augmentation to preserve spatial structure for reconstruction.

    Args:
        image_size: Target image size
        num_channels: Number of image channels (1 or 3)

    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]

    # Normalize
    if num_channels == 1:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return transforms.Compose(transform_list)


def denormalize(tensor, num_channels: int = 3):
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        num_channels: Number of channels

    Returns:
        Denormalized tensor in [0, 1] range
    """
    if num_channels == 1:
        mean = [0.5]
        std = [0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Handle batch dimension
    if tensor.dim() == 4:
        mean = tensor.new_tensor(mean).view(1, -1, 1, 1)
        std = tensor.new_tensor(std).view(1, -1, 1, 1)
    else:
        mean = tensor.new_tensor(mean).view(-1, 1, 1)
        std = tensor.new_tensor(std).view(-1, 1, 1)

    return tensor * std + mean
