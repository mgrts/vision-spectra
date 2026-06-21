"""Tests for the capacity×complexity study wiring: the sweep grid, head/bulk
truncation, and the gradient-alignment probe added to the spectral runner."""

from __future__ import annotations

import numpy as np
import pytest
import timm
import torch

from vision_spectra.experiments.run_spectral_analysis import (
    ScenarioConfig,
    ScenarioType,
    build_study_configs,
    create_model_for_scenario,
    record_gradient_alignment,
    run_truncation_analysis,
)
from vision_spectra.metrics.tail_truncation import truncate_weight_matrix


class TestStudyGrid:
    """build_study_configs and the architectural invariants of every cell."""

    def test_tier_sizes_monotone(self) -> None:
        n1 = len(build_study_configs(1))
        n2 = len(build_study_configs(2))
        n3 = len(build_study_configs(3))
        assert n1 == 8  # 4 widths × {synthetic, pathmnist}
        assert n2 > n1  # + depth sweep + blood/derma
        assert n3 > n2  # + A-F

    def test_every_cell_has_valid_head_count(self) -> None:
        # num_heads = max(1, embed//32) must divide embed_dim for timm to build the ViT.
        for cfg in build_study_configs(3):
            num_heads = max(1, cfg.embed_dim // 32)
            assert cfg.embed_dim % num_heads == 0, f"{cfg.name}: {cfg.embed_dim}/{num_heads}"

    def test_enum_configs_preserve_mlflow_contract(self) -> None:
        # A-F MUST map to experiment name spectral_scenario_{A-F} (figures contract).
        cfg_a = ScenarioConfig(
            scenario=ScenarioType.A_EXPRESSIVE_SIMPLE,
            model_name="m",
            embed_dim=192,
            depth=6,
            dataset_name="synthetic",
            num_samples=10,
            num_classes=3,
            epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seeds=[0],
            log_epochs=[0],
            description="d",
        )
        assert cfg_a.name == "scenario_A"
        assert cfg_a.scenario_label == "A"

    def test_string_tag_configs(self) -> None:
        cfg = ScenarioConfig(
            scenario="w096_path",
            model_name="m",
            embed_dim=96,
            depth=6,
            dataset_name="pathmnist",
            num_samples=None,
            num_classes=9,
            epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seeds=[0],
            log_epochs=[0],
            description="d",
        )
        assert cfg.name == "w096_path"
        assert cfg.scenario_label == "w096_path"

    @pytest.mark.parametrize("embed,depth", [(48, 2), (96, 1)])
    def test_model_builds_with_asserted_depth(self, embed: int, depth: int) -> None:
        cfg = ScenarioConfig(
            scenario=f"w{embed}_test",
            model_name="m",
            embed_dim=embed,
            depth=depth,
            dataset_name="synthetic",
            num_samples=10,
            num_classes=3,
            epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seeds=[0],
            log_epochs=[0],
            description="d",
        )
        model = create_model_for_scenario(cfg, torch.device("cpu"))
        assert len(model.blocks) == depth


class TestHeadVsBulkTruncation:
    """truncate_weight_matrix mode semantics (SVD returns descending order)."""

    def _matrix_with_known_spectrum(self) -> np.ndarray:
        # Build W = U diag([10,3,1,0.3,0.1]) Vᵀ with orthonormal U, V (10x5).
        rng = np.random.RandomState(0)
        u, _ = np.linalg.qr(rng.randn(10, 5))
        v, _ = np.linalg.qr(rng.randn(5, 5))
        s = np.array([10.0, 3.0, 1.0, 0.3, 0.1])
        return (u * s) @ v.T

    def test_bulk_keeps_largest(self) -> None:
        w = self._matrix_with_known_spectrum()
        # retain 2/5 of SVs → keep the two LARGEST (10, 3).
        trunc, info = truncate_weight_matrix(w, retention_ratio=0.4, mode="bulk")
        s_new = np.linalg.svd(trunc, compute_uv=False)
        kept = np.sort(s_new[s_new > 1e-8])[::-1]
        assert np.allclose(kept, [10.0, 3.0], atol=1e-6)
        # energy_retained = (100+9)/(100+9+1+.09+.01) ≈ 0.99
        assert info["energy_retained"] > 0.98

    def test_head_drops_largest(self) -> None:
        w = self._matrix_with_known_spectrum()
        # retain 2/5 of SVs in HEAD mode → keep the two SMALLEST (0.3, 0.1),
        # zero the three LARGEST outliers (10, 3, 1).
        trunc, info = truncate_weight_matrix(w, retention_ratio=0.4, mode="head")
        s_new = np.linalg.svd(trunc, compute_uv=False)
        kept = np.sort(s_new[s_new > 1e-8])[::-1]
        assert np.allclose(kept, [0.3, 0.1], atol=1e-6)
        # almost no energy retained — the heavy-tail outliers carried it.
        assert info["energy_retained"] < 0.01


class TestAlignmentAndTruncationOnModel:
    """The two probes run on a real (tiny) ViT and produce finite, sane outputs."""

    def _tiny_model(self) -> torch.nn.Module:
        return timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=3,
            in_chans=3,
            img_size=28,
            patch_size=4,
            embed_dim=32,
            depth=1,
            num_heads=1,
        )

    def _loader(self):
        x = torch.randn(16, 3, 28, 28)
        y = torch.randint(0, 3, (16,))
        ds = torch.utils.data.TensorDataset(x, y)
        return torch.utils.data.DataLoader(ds, batch_size=8)

    def test_alignment_returns_finite_and_zeroes_grads(self) -> None:
        model = self._tiny_model()
        model.train()
        probe = next(iter(self._loader()))
        agg = record_gradient_alignment(
            model, probe, torch.nn.CrossEntropyLoss(), torch.device("cpu")
        )
        assert set(agg) >= {"cos_sim_mean", "fraction_aligned", "angle_mean"}
        assert np.isfinite(agg["cos_sim_mean"])
        assert -1.0 <= agg["cos_sim_mean"] <= 1.0
        # grads were zeroed; training mode restored.
        assert model.training
        assert all(p.grad is None or torch.all(p.grad == 0) for p in model.parameters())

    def test_truncation_summary_has_both_modes(self, tmp_path) -> None:
        import mlflow

        mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
        model = self._tiny_model()
        with mlflow.start_run(nested=False):
            summary = run_truncation_analysis(
                model, self._loader(), torch.device("cpu"), max_batches=2
            )
        assert set(summary) == {"bulk", "head"}
        for mode in ("bulk", "head"):
            assert "results" in summary[mode] and "analysis" in summary[mode]
            assert all(r["mode"] == mode for r in summary[mode]["results"])
