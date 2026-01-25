"""
Vision Transformer (ViT) classifier using timm.

Provides a clean wrapper around timm models with:
- Custom model creation for small images (28x28)
- Easy weight extraction for spectral analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import timm
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from vision_spectra.settings import ModelConfig


class ViTClassifier(nn.Module):
    """
    ViT classifier wrapper.

    Wraps a timm ViT model and provides convenient access to:
    - Forward pass for classification
    - Feature extraction before classification head
    - Weight matrices for spectral analysis

    Supports configurable embed_dim and depth for expressivity control experiments.
    """

    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        num_classes: int = 10,
        num_channels: int = 3,
        image_size: int = 28,
        pretrained: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        embed_dim: int | None = None,
        depth: int | None = None,
        num_heads: int | None = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.image_size = image_size

        # Build kwargs for model creation
        model_kwargs = {
            "pretrained": pretrained,
            "num_classes": num_classes,
            "in_chans": num_channels,
            "img_size": image_size,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
        }

        # Add optional architecture customization for expressivity control
        if embed_dim is not None:
            model_kwargs["embed_dim"] = embed_dim
        if depth is not None:
            model_kwargs["depth"] = depth
        if num_heads is not None:
            model_kwargs["num_heads"] = num_heads
        elif embed_dim is not None:
            # Auto-calculate num_heads based on embed_dim
            model_kwargs["num_heads"] = max(1, embed_dim // 32)

        # Create model with custom configuration
        self.encoder = timm.create_model(model_name, **model_kwargs)

        # Store model config
        self.embed_dim = self.encoder.embed_dim
        self.num_heads = (
            self.encoder.blocks[0].attn.num_heads if hasattr(self.encoder, "blocks") else 4
        )
        self.num_blocks = len(self.encoder.blocks) if hasattr(self.encoder, "blocks") else 0

        # Extract patch size from patch_embed
        patch_embed = getattr(self.encoder, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "patch_size"):
            ps = patch_embed.patch_size
            if isinstance(ps, tuple | list):
                self.patch_size = ps[0]
            else:
                self.patch_size = int(ps)
        else:
            self.patch_size = 16  # Default for vit_tiny_patch16_224

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Logits [B, num_classes]
        """
        return self.encoder(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification head.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Features [B, embed_dim] (pooled) or [B, num_patches+1, embed_dim] (sequence)
        """
        return self.encoder.forward_features(x)

    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch embeddings including CLS token.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Patch embeddings [B, num_patches+1, embed_dim]
        """
        return self.encoder.patch_embed(x)

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Get attention weights from all blocks.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            List of attention weights [B, num_heads, num_patches+1, num_patches+1]
        """
        attention_weights = []

        # Get patch embeddings
        x = self.encoder.patch_embed(x)

        # Add CLS token
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Add position embeddings
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)

        # Forward through blocks and collect attention
        for blk in self.encoder.blocks:
            # Get attention weights
            B, N, C = x.shape
            qkv = (
                blk.attn.qkv(x)
                .reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv.unbind(0)

            attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
            attn = attn.softmax(dim=-1)
            attention_weights.append(attn.detach())

            # Continue forward pass
            x = blk(x)

        return attention_weights


def create_vit_classifier(
    config: ModelConfig,
    num_classes: int,
    num_channels: int = 3,
    image_size: int = 28,
    embed_dim: int | None = None,
    depth: int | None = None,
    num_heads: int | None = None,
) -> ViTClassifier:
    """
    Create a ViT classifier from config.

    Args:
        config: Model configuration
        num_classes: Number of output classes
        num_channels: Number of input channels
        image_size: Input image size
        embed_dim: Override embedding dimension (for expressivity control)
        depth: Override number of transformer blocks (for expressivity control)
        num_heads: Override number of attention heads

    Returns:
        ViTClassifier instance
    """
    return ViTClassifier(
        model_name=config.name,
        num_classes=num_classes,
        num_channels=num_channels,
        image_size=image_size,
        pretrained=config.pretrained,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
    )


# Available ViT models suitable for small images (with dynamic_img_size=True)
SMALL_IMAGE_MODELS = [
    "vit_tiny_patch16_224",  # Tiny ViT - works with small images via dynamic_img_size
    "vit_small_patch16_224",  # Small ViT
    "vit_base_patch16_224",  # Base ViT
    "deit_tiny_patch16_224",  # DeiT Tiny
    "deit_small_patch16_224",  # DeiT Small
]


def get_available_models() -> list[str]:
    """Get list of available model names."""
    return SMALL_IMAGE_MODELS
