"""
Multitask Vision Transformer.

Combines classification and MIM in a single model for joint training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vision_spectra.models.mim import MIMDecoder
from vision_spectra.models.vit import ViTClassifier


class MultitaskViT(nn.Module):
    """
    Multitask Vision Transformer for joint classification and MIM.

    Supports three forward modes:
    1. Classification only
    2. MIM only
    3. Joint multitask (both losses)
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
        self.num_classes = encoder.num_classes
        self.image_size = encoder.image_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # MIM decoder
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
        """Convert images to patches."""
        from einops import rearrange

        p = self.patch_size
        x = rearrange(imgs, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        return x

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform random masking on patch embeddings."""
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification only.

        Args:
            x: Input images [B, C, H, W]

        Returns:
            logits: Classification logits [B, num_classes]
        """
        return self.encoder(x)

    def forward_mim(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for MIM only.

        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Mask ratio (uses default if None)

        Returns:
            loss: MIM loss
            pred: Predicted patches
            mask: Binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Encode with masking
        latent, mask, ids_restore = self._forward_encoder_masked(x, mask_ratio)

        # Decode
        pred = self.decoder(latent, ids_restore, self.decoder_pos_embed)

        # Compute loss
        loss = self._compute_mim_loss(x, pred, mask)

        return loss, pred, mask

    def forward_multitask(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for joint classification and MIM.

        Uses full image for classification, masked for MIM.

        Args:
            x: Input images [B, C, H, W]
            mask_ratio: Mask ratio for MIM

        Returns:
            logits: Classification logits [B, num_classes]
            mim_loss: MIM loss
            pred: Predicted patches
            mask: Binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # Classification uses full image (no masking)
        logits = self.encoder(x)

        # MIM uses masked encoding
        latent, mask, ids_restore = self._forward_encoder_masked(x, mask_ratio)
        pred = self.decoder(latent, ids_restore, self.decoder_pos_embed)
        mim_loss = self._compute_mim_loss(x, pred, mask)

        return logits, mim_loss, pred, mask

    def _forward_encoder_masked(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward through encoder with masking."""
        # Get patch embeddings
        x = self.encoder.encoder.patch_embed(x)

        # Add position embeddings (skip CLS)
        pos_embed = self.encoder.encoder.pos_embed[:, 1:, :]
        x = x + pos_embed

        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Add CLS token
        cls_token = self.encoder.encoder.cls_token + self.encoder.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer blocks
        x = self.encoder.encoder.pos_drop(x)
        for blk in self.encoder.encoder.blocks:
            x = blk(x)
        x = self.encoder.encoder.norm(x)

        # Remove CLS token
        latent = x[:, 1:, :]

        return latent, mask, ids_restore

    def _compute_mim_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MIM reconstruction loss."""
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

        return loss

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "classification",
        mask_ratio: float | None = None,
    ):
        """
        Unified forward pass.

        Args:
            x: Input images
            mode: One of "classification", "mim", "multitask"
            mask_ratio: Mask ratio for MIM modes

        Returns:
            Depends on mode
        """
        if mode == "classification":
            return self.forward_classification(x)
        elif mode == "mim":
            return self.forward_mim(x, mask_ratio)
        elif mode == "multitask":
            return self.forward_multitask(x, mask_ratio)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'classification', 'mim', or 'multitask'")
