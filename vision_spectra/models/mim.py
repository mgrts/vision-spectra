"""
Masked Image Modeling (MIM) components.

Implements a decoder for reconstructing masked patches,
following the MAE (Masked Autoencoder) approach.

Reference:
    He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

if TYPE_CHECKING:
    from vision_spectra.models.vit import ViTClassifier


class MIMDecoder(nn.Module):
    """
    Lightweight decoder for masked image modeling.

    Takes encoded visible patches and reconstructs the full image
    (or just the masked patches for efficiency).
    """

    def __init__(
        self,
        encoder_embed_dim: int = 192,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        patch_size: int = 4,
        num_channels: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Project from encoder to decoder dimension
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Decoder transformer blocks
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Predict pixel values for each patch
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * num_channels,
            bias=True,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.normal_(self.mask_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        pos_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode masked patches.

        Args:
            x: Encoded visible patches [B, N_visible, encoder_embed_dim]
            ids_restore: Indices to restore full sequence [B, N_total]
            pos_embed: Position embeddings for decoder [B, N_total+1, decoder_embed_dim]

        Returns:
            Reconstructed patches [B, N_total, patch_size^2 * C]
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Append mask tokens
        B, N_visible, D = x.shape
        N_total = ids_restore.shape[1]
        N_mask = N_total - N_visible

        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        # Concatenate and restore order
        x_full = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle to original positions
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, D)
        x_full = torch.gather(x_full, dim=1, index=ids_restore_expanded)

        # Add position embeddings (skip CLS token position)
        if pos_embed is not None:
            x_full = x_full + pos_embed[:, 1:, :]  # Skip CLS

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x_full = blk(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict pixel values
        x_full = self.decoder_pred(x_full)

        return x_full


class TransformerBlock(nn.Module):
    """Simple transformer block for decoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class MIMModel(nn.Module):
    """
    Complete Masked Image Modeling model.

    Combines ViT encoder with MIM decoder for self-supervised pretraining.
    """

    def __init__(
        self,
        encoder: ViTClassifier,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 2,
        decoder_num_heads: int = 4,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Get encoder parameters
        self.patch_size = encoder.patch_size
        self.num_channels = encoder.num_channels
        self.embed_dim = encoder.embed_dim

        # Calculate number of patches
        self.image_size = encoder.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # Create decoder
        self.decoder = MIMDecoder(
            encoder_embed_dim=self.embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
        )

        # Decoder position embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim)
        )
        nn.init.normal_(self.decoder_pos_embed, std=0.02)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.

        Args:
            imgs: [B, C, H, W]

        Returns:
            patches: [B, num_patches, patch_size^2 * C]
        """
        p = self.patch_size

        x = rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.

        Args:
            x: [B, num_patches, patch_size^2 * C]

        Returns:
            imgs: [B, C, H, W]
        """
        p = self.patch_size
        h = w = self.image_size // p
        c = self.num_channels

        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p, p2=p, c=c)
        return x

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform random masking.

        Args:
            x: Patch embeddings [B, N, D] (without CLS token)
            mask_ratio: Fraction of patches to mask

        Returns:
            x_masked: Visible patches [B, N_visible, D]
            mask: Binary mask [B, N] (1 = masked, 0 = visible)
            ids_restore: Indices to restore order [B, N]
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)

        # Sort noise to get shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep patches
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate binary mask (1 = masked)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder with masking.

        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Fraction of patches to mask

        Returns:
            latent: Encoded visible patches [B, N_visible, D]
            mask: Binary mask [B, N]
            ids_restore: Indices to restore order [B, N]
        """
        # Get patch embeddings (without CLS token)
        x = self.encoder.encoder.patch_embed(x)

        # Add position embeddings
        pos_embed = self.encoder.encoder.pos_embed[:, 1:, :]  # Skip CLS
        x = x + pos_embed

        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Add CLS token
        cls_token = self.encoder.encoder.cls_token + self.encoder.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        x = self.encoder.encoder.pos_drop(x)
        for blk in self.encoder.encoder.blocks:
            x = blk(x)
        x = self.encoder.encoder.norm(x)

        # Remove CLS token for decoder
        latent = x[:, 1:, :]

        return latent, mask, ids_restore

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            latent: Encoded visible patches [B, N_visible, D]
            ids_restore: Indices to restore order [B, N]

        Returns:
            pred: Reconstructed patches [B, N, patch_size^2 * C]
        """
        pred = self.decoder(latent, ids_restore, self.decoder_pos_embed)
        return pred

    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches.

        Args:
            imgs: Original images [B, C, H, W]
            pred: Predicted patches [B, N, patch_size^2 * C]
            mask: Binary mask [B, N] (1 = masked)

        Returns:
            loss: Scalar loss value
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            # Normalize target per patch
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss on masked patches only
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]

        # Average over masked patches
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            imgs: Input images [B, C, H, W]
            mask_ratio: Override mask ratio (uses self.mask_ratio if None)

        Returns:
            loss: Reconstruction loss
            pred: Predicted patches [B, N, patch_size^2 * C]
            mask: Binary mask [B, N]
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask
