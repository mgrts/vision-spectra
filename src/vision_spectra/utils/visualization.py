"""
Visualization utilities for saving prediction examples to MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def denormalize_image(tensor: torch.Tensor, num_channels: int = 3) -> torch.Tensor:
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

    return torch.clamp(tensor * std + mean, 0, 1)


def save_prediction_examples(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: Path,
    num_examples: int = 16,
    num_channels: int = 3,
    class_names: list[str] | None = None,
    device: torch.device | None = None,
) -> list[Path]:
    """
    Save example predictions as image grid to disk.

    Args:
        model: Trained model
        dataloader: DataLoader to get examples from
        save_dir: Directory to save images
        num_examples: Number of examples to save
        num_channels: Number of image channels
        class_names: Optional list of class names for labels
        device: Device to run inference on

    Returns:
        List of paths to saved images
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Collect examples
    images_list = []
    labels_list = []
    preds_list = []
    probs_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            images_list.append(images.cpu())
            labels_list.append(labels.cpu())
            preds_list.append(preds.cpu())
            probs_list.append(probs.cpu())

            if sum(len(x) for x in images_list) >= num_examples:
                break

    # Concatenate all batches
    all_images = torch.cat(images_list, dim=0)[:num_examples]
    all_labels = torch.cat(labels_list, dim=0)[:num_examples]
    all_preds = torch.cat(preds_list, dim=0)[:num_examples]
    all_probs = torch.cat(probs_list, dim=0)[:num_examples]

    # Denormalize images
    all_images = denormalize_image(all_images, num_channels)

    saved_paths = []

    # Save grid of all predictions
    grid_path = save_dir / "prediction_examples.png"
    _save_prediction_grid(
        all_images,
        all_labels,
        all_preds,
        all_probs,
        grid_path,
        num_channels,
        class_names,
    )
    saved_paths.append(grid_path)

    # Save separate grids for correct and incorrect predictions
    correct_mask = all_preds == all_labels
    incorrect_mask = ~correct_mask

    if correct_mask.sum() > 0:
        correct_path = save_dir / "correct_predictions.png"
        _save_prediction_grid(
            all_images[correct_mask],
            all_labels[correct_mask],
            all_preds[correct_mask],
            all_probs[correct_mask],
            correct_path,
            num_channels,
            class_names,
            title="Correct Predictions",
        )
        saved_paths.append(correct_path)

    if incorrect_mask.sum() > 0:
        incorrect_path = save_dir / "incorrect_predictions.png"
        _save_prediction_grid(
            all_images[incorrect_mask],
            all_labels[incorrect_mask],
            all_preds[incorrect_mask],
            all_probs[incorrect_mask],
            incorrect_path,
            num_channels,
            class_names,
            title="Incorrect Predictions",
        )
        saved_paths.append(incorrect_path)

    # Save input data examples (first batch without predictions)
    input_path = save_dir / "input_examples.png"
    _save_input_grid(all_images, all_labels, input_path, num_channels, class_names)
    saved_paths.append(input_path)

    return saved_paths


def _save_prediction_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    probs: torch.Tensor,
    save_path: Path,
    num_channels: int = 3,
    class_names: list[str] | None = None,
    title: str = "Model Predictions",
) -> None:
    """Save a grid of images with predictions."""
    n = len(images)
    if n == 0:
        return

    # Determine grid size
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        if idx < n:
            img = images[idx]
            label = labels[idx].item()
            pred = preds[idx].item()
            prob = probs[idx, pred].item()

            # Convert to numpy for plotting
            if num_channels == 1:
                img_np = img.squeeze(0).numpy()
                ax.imshow(img_np, cmap="gray")
            else:
                img_np = img.permute(1, 2, 0).numpy()
                ax.imshow(img_np)

            # Get class names
            label_name = class_names[label] if class_names else str(label)
            pred_name = class_names[pred] if class_names else str(pred)

            # Color based on correctness
            color = "green" if label == pred else "red"
            ax.set_title(
                f"True: {label_name}\nPred: {pred_name} ({prob:.1%})",
                fontsize=9,
                color=color,
            )
        else:
            ax.axis("off")
            continue

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_input_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    num_channels: int = 3,
    class_names: list[str] | None = None,
) -> None:
    """Save a grid of input images with their labels."""
    n = len(images)
    if n == 0:
        return

    # Determine grid size
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows))
    fig.suptitle("Input Data Examples", fontsize=14, fontweight="bold")

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        if idx < n:
            img = images[idx]
            label = labels[idx].item()

            # Convert to numpy for plotting
            if num_channels == 1:
                img_np = img.squeeze(0).numpy()
                ax.imshow(img_np, cmap="gray")
            else:
                img_np = img.permute(1, 2, 0).numpy()
                ax.imshow(img_np)

            # Get class name
            label_name = class_names[label] if class_names else str(label)
            ax.set_title(f"Label: {label_name}", fontsize=10)
        else:
            ax.axis("off")
            continue

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mim_examples(
    model: nn.Module,
    dataloader: DataLoader,
    save_dir: Path,
    num_examples: int = 8,
    num_channels: int = 3,
    device: torch.device | None = None,
) -> list[Path]:
    """
    Save MIM (Masked Image Modeling) examples showing original, masked, and reconstructed images.

    Args:
        model: MIM model (must have forward method returning loss, pred, mask)
        dataloader: DataLoader to get examples from
        save_dir: Directory to save images
        num_examples: Number of examples to save
        num_channels: Number of image channels
        device: Device to run inference on

    Returns:
        List of paths to saved images
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Collect examples
    originals = []
    reconstructions = []
    masks = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            # Get MIM outputs
            try:
                loss, pred, mask = model(images)
            except Exception:
                # Model might not support MIM forward
                return []

            # Get reconstructed images from model
            try:
                reconstructed = model.unpatchify(pred)
            except Exception:
                # Try to reconstruct manually if unpatchify not available
                return []

            originals.append(images.cpu())
            reconstructions.append(reconstructed.cpu())
            masks.append(mask.cpu())

            if sum(len(x) for x in originals) >= num_examples:
                break

    if not originals:
        return []

    # Concatenate all batches
    all_originals = torch.cat(originals, dim=0)[:num_examples]
    all_reconstructions = torch.cat(reconstructions, dim=0)[:num_examples]
    all_masks = torch.cat(masks, dim=0)[:num_examples]

    # Denormalize images
    all_originals = denormalize_image(all_originals, num_channels)
    all_reconstructions = denormalize_image(all_reconstructions, num_channels)

    saved_paths = []

    # Save MIM comparison grid
    mim_path = save_dir / "mim_examples.png"
    _save_mim_grid(
        all_originals,
        all_reconstructions,
        all_masks,
        mim_path,
        num_channels,
        model,
    )
    saved_paths.append(mim_path)

    return saved_paths


def _save_mim_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    masks: torch.Tensor,
    save_path: Path,
    num_channels: int = 3,
    model: nn.Module | None = None,
) -> None:
    """
    Save a grid showing original, masked, and reconstructed images side by side.

    Each row shows: Original | Masked | Reconstructed | Difference
    """
    n = len(originals)
    if n == 0:
        return

    # Limit to 8 examples max for readability
    n = min(n, 8)

    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    fig.suptitle(
        "MIM: Original → Masked → Reconstructed → Difference", fontsize=14, fontweight="bold"
    )

    if n == 1:
        axes = axes.reshape(1, -1)

    # Get patch size and image size from model if available
    patch_size = getattr(model, "patch_size", 16) if model else 16
    image_size = originals.shape[-1]

    for idx in range(n):
        orig = originals[idx]
        recon = reconstructions[idx]
        mask = masks[idx]

        # Create masked visualization
        masked_img = _create_masked_image(orig, mask, patch_size, image_size, num_channels)

        # Compute difference (only in masked regions for visibility)
        diff = torch.abs(orig - recon)

        # Convert to numpy
        if num_channels == 1:
            orig_np = orig.squeeze(0).numpy()
            masked_np = masked_img.squeeze(0).numpy()
            recon_np = recon.squeeze(0).numpy()
            diff_np = diff.squeeze(0).numpy()
            cmap = "gray"
        else:
            orig_np = orig.permute(1, 2, 0).numpy()
            masked_np = masked_img.permute(1, 2, 0).numpy()
            recon_np = recon.permute(1, 2, 0).numpy()
            diff_np = diff.permute(1, 2, 0).numpy()
            cmap = None

        # Plot original
        axes[idx, 0].imshow(orig_np, cmap=cmap)
        axes[idx, 0].set_title("Original" if idx == 0 else "", fontsize=10)
        axes[idx, 0].axis("off")

        # Plot masked
        axes[idx, 1].imshow(masked_np, cmap=cmap)
        axes[idx, 1].set_title("Masked" if idx == 0 else "", fontsize=10)
        axes[idx, 1].axis("off")

        # Plot reconstructed
        axes[idx, 2].imshow(recon_np, cmap=cmap)
        axes[idx, 2].set_title("Reconstructed" if idx == 0 else "", fontsize=10)
        axes[idx, 2].axis("off")

        # Plot difference (use hot colormap for visibility)
        if num_channels == 1:
            axes[idx, 3].imshow(diff_np, cmap="hot", vmin=0, vmax=0.5)
        else:
            # For RGB, show mean absolute difference
            diff_mean = diff_np.mean(axis=-1)
            axes[idx, 3].imshow(diff_mean, cmap="hot", vmin=0, vmax=0.5)
        axes[idx, 3].set_title("Difference" if idx == 0 else "", fontsize=10)
        axes[idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _create_masked_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    image_size: int,
    num_channels: int,
) -> torch.Tensor:
    """
    Create a visualization of the masked image.

    Args:
        image: Original image [C, H, W]
        mask: Binary mask [num_patches] where 1 = masked
        patch_size: Size of each patch
        image_size: Size of the image
        num_channels: Number of channels

    Returns:
        Masked image with gray patches where masked
    """
    masked_img = image.clone()
    num_patches_per_side = image_size // patch_size

    # Reshape mask to 2D grid
    mask_2d = mask.reshape(num_patches_per_side, num_patches_per_side)

    # Apply gray color to masked patches
    gray_value = 0.5

    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            if mask_2d[i, j] == 1:  # Masked patch
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size
                masked_img[:, y_start:y_end, x_start:x_end] = gray_value

    return masked_img
