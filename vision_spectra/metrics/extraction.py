"""
Weight matrix extraction from transformer models.

Provides utilities for extracting Q/K/V projection weights,
MLP weights, and other matrices for spectral analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class WeightInfo:
    """Information about an extracted weight matrix."""

    name: str
    layer_idx: int | None
    matrix_type: str  # 'qkv', 'proj', 'mlp', 'patch_embed', etc.
    weight: np.ndarray
    shape: tuple[int, ...]

    def __repr__(self) -> str:
        return f"WeightInfo(name='{self.name}', type='{self.matrix_type}', shape={self.shape})"


def extract_qkv_weights(
    model: nn.Module,
    layer_patterns: list[str] | None = None,
) -> list[WeightInfo]:
    """
    Extract Q, K, V projection weights from transformer blocks.

    Args:
        model: Transformer model (ViT or similar)
        layer_patterns: List of layer name patterns to match (e.g., ["blocks.0", "blocks.2"])
                       If None, extracts from all blocks.

    Returns:
        List of WeightInfo for each extracted matrix
    """
    weights: list[WeightInfo] = []

    for name, module in model.named_modules():
        # Skip if doesn't match pattern
        if layer_patterns and not any(pat in name for pat in layer_patterns):
            continue

        # Handle combined QKV weight (common in timm)
        if hasattr(module, "qkv") and hasattr(module.qkv, "weight"):
            qkv_weight = module.qkv.weight.detach().cpu().numpy()

            # QKV is concatenated: [3 * embed_dim, embed_dim]
            embed_dim = qkv_weight.shape[1]
            q_weight = qkv_weight[:embed_dim]
            k_weight = qkv_weight[embed_dim : 2 * embed_dim]
            v_weight = qkv_weight[2 * embed_dim :]

            # Extract layer index from name
            layer_idx = _extract_layer_idx(name)

            weights.extend(
                [
                    WeightInfo(
                        name=f"{name}.qkv.q",
                        layer_idx=layer_idx,
                        matrix_type="q",
                        weight=q_weight,
                        shape=q_weight.shape,
                    ),
                    WeightInfo(
                        name=f"{name}.qkv.k",
                        layer_idx=layer_idx,
                        matrix_type="k",
                        weight=k_weight,
                        shape=k_weight.shape,
                    ),
                    WeightInfo(
                        name=f"{name}.qkv.v",
                        layer_idx=layer_idx,
                        matrix_type="v",
                        weight=v_weight,
                        shape=v_weight.shape,
                    ),
                ]
            )

        # Handle separate Q, K, V weights
        elif hasattr(module, "q_proj") and hasattr(module.q_proj, "weight"):
            layer_idx = _extract_layer_idx(name)

            for proj_name, proj_type in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                if hasattr(module, proj_name):
                    proj_module = getattr(module, proj_name)
                    if hasattr(proj_module, "weight"):
                        weight = proj_module.weight.detach().cpu().numpy()
                        weights.append(
                            WeightInfo(
                                name=f"{name}.{proj_name}",
                                layer_idx=layer_idx,
                                matrix_type=proj_type,
                                weight=weight,
                                shape=weight.shape,
                            )
                        )

    return weights


def extract_attention_weights(
    model: nn.Module,
    layer_patterns: list[str] | None = None,
) -> list[WeightInfo]:
    """
    Extract attention output projection weights.

    Args:
        model: Transformer model
        layer_patterns: Layer name patterns to match

    Returns:
        List of WeightInfo for projection matrices
    """
    weights: list[WeightInfo] = []

    for name, module in model.named_modules():
        if layer_patterns and not any(pat in name for pat in layer_patterns):
            continue

        # Output projection (timm uses 'proj')
        # Make sure this is attention proj, not patch embed proj
        if (
            hasattr(module, "proj")
            and hasattr(module.proj, "weight")
            and ("attn" in name.lower() or "attention" in name.lower())
        ):
            weight = module.proj.weight.detach().cpu().numpy()
            layer_idx = _extract_layer_idx(name)

            weights.append(
                WeightInfo(
                    name=f"{name}.proj",
                    layer_idx=layer_idx,
                    matrix_type="attn_proj",
                    weight=weight,
                    shape=weight.shape,
                )
            )

    return weights


def extract_mlp_weights(
    model: nn.Module,
    layer_patterns: list[str] | None = None,
) -> list[WeightInfo]:
    """
    Extract MLP (FFN) weights from transformer blocks.

    Args:
        model: Transformer model
        layer_patterns: Layer name patterns to match

    Returns:
        List of WeightInfo for MLP matrices
    """
    weights: list[WeightInfo] = []

    for name, module in model.named_modules():
        if layer_patterns and not any(pat in name for pat in layer_patterns):
            continue

        # MLP layers (timm uses 'mlp.fc1', 'mlp.fc2')
        if (
            ("mlp" in name.lower() or "ffn" in name.lower())
            and hasattr(module, "weight")
            and isinstance(module.weight, torch.Tensor)
        ):
            weight = module.weight.detach().cpu().numpy()
            layer_idx = _extract_layer_idx(name)

            # Determine if it's the first or second MLP layer
            if "fc1" in name or "0" in name.split(".")[-1]:
                mlp_type = "mlp_up"
            elif "fc2" in name or "2" in name.split(".")[-1]:
                mlp_type = "mlp_down"
            else:
                mlp_type = "mlp"

            weights.append(
                WeightInfo(
                    name=name,
                    layer_idx=layer_idx,
                    matrix_type=mlp_type,
                    weight=weight,
                    shape=weight.shape,
                )
            )

    return weights


def extract_patch_embed_weights(model: nn.Module) -> list[WeightInfo]:
    """
    Extract patch embedding weights.

    Args:
        model: Transformer model with patch embedding

    Returns:
        List of WeightInfo for patch embedding matrices
    """
    weights: list[WeightInfo] = []

    for name, module in model.named_modules():
        if (
            "patch_embed" in name.lower()
            and hasattr(module, "proj")
            and hasattr(module.proj, "weight")
        ):
            weight = module.proj.weight.detach().cpu().numpy()

            # Reshape conv weight [out, in, h, w] -> [out, in*h*w]
            if weight.ndim == 4:
                weight = weight.reshape(weight.shape[0], -1)

            weights.append(
                WeightInfo(
                    name=f"{name}.proj",
                    layer_idx=None,
                    matrix_type="patch_embed",
                    weight=weight,
                    shape=weight.shape,
                )
            )

    return weights


def extract_all_weights(
    model: nn.Module,
    layer_patterns: list[str] | None = None,
    include_qkv: bool = True,
    include_proj: bool = True,
    include_mlp: bool = False,
    include_patch_embed: bool = True,
) -> list[WeightInfo]:
    """
    Extract all specified weight matrices from a model.

    Args:
        model: Transformer model
        layer_patterns: Layer patterns to match
        include_qkv: Include Q/K/V projection weights
        include_proj: Include attention output projection
        include_mlp: Include MLP weights
        include_patch_embed: Include patch embedding weights

    Returns:
        List of all extracted WeightInfo
    """
    weights: list[WeightInfo] = []

    if include_qkv:
        weights.extend(extract_qkv_weights(model, layer_patterns))

    if include_proj:
        weights.extend(extract_attention_weights(model, layer_patterns))

    if include_mlp:
        weights.extend(extract_mlp_weights(model, layer_patterns))

    if include_patch_embed:
        weights.extend(extract_patch_embed_weights(model))

    return weights


def _extract_layer_idx(name: str) -> int | None:
    """Extract layer index from module name."""
    # Match patterns like 'blocks.0', 'layer.1', 'encoder.layers.2'
    match = re.search(r"(?:blocks|layers?|encoder\.layer)\.(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def group_weights_by_layer(
    weights: list[WeightInfo],
) -> dict[int | None, list[WeightInfo]]:
    """
    Group weight infos by layer index.

    Args:
        weights: List of WeightInfo objects

    Returns:
        Dictionary mapping layer_idx to list of weights
    """
    grouped: dict[int | None, list[WeightInfo]] = {}

    for w in weights:
        if w.layer_idx not in grouped:
            grouped[w.layer_idx] = []
        grouped[w.layer_idx].append(w)

    return grouped


def group_weights_by_type(
    weights: list[WeightInfo],
) -> dict[str, list[WeightInfo]]:
    """
    Group weight infos by matrix type.

    Args:
        weights: List of WeightInfo objects

    Returns:
        Dictionary mapping matrix_type to list of weights
    """
    grouped: dict[str, list[WeightInfo]] = {}

    for w in weights:
        if w.matrix_type not in grouped:
            grouped[w.matrix_type] = []
        grouped[w.matrix_type].append(w)

    return grouped
