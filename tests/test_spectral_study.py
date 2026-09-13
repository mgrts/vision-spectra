"""Tests for the capacity×complexity study wiring: the sweep grid, head/bulk
truncation, and the gradient-alignment probe added to the spectral runner."""

from __future__ import annotations

import numpy as np
import pytest
import timm
import torch
from pydantic import ValidationError

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
        assert set(summary) == {"bulk", "head", "head_drop"}
        for mode in ("bulk", "head"):
            assert "results" in summary[mode] and "analysis" in summary[mode]
            assert all(r["mode"] == mode for r in summary[mode]["results"])


class TestSubspaceAlignmentProbe:
    """The subspace-resolved probe has full [-1, 1] range and correct signs/baselines."""

    def _basis(self, m: int = 64, n: int = 48):
        rng = np.random.default_rng(0)
        w = rng.normal(size=(m, n))
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        return w, u, vt

    def test_tail_uniform_gradient_gives_cos_tail_one(self) -> None:
        from vision_spectra.metrics.gradient_alignment import head_k_for_rank, subspace_alignment

        w, u, vt = self._basis()
        k = head_k_for_rank(min(w.shape))
        g = u[:, k:] @ vt[k:]  # +U_t V_tᵀ: the step −ηG shrinks every tail σ
        out = subspace_alignment(g, w)
        assert out["cos_tail"] == pytest.approx(1.0, abs=1e-9)
        assert out["cos_head"] == pytest.approx(0.0, abs=1e-9)
        assert out["frac_tail_shrinking"] == pytest.approx(1.0)
        assert out["k"] == k and out["rank"] == min(w.shape)

    def test_head_growing_gradient_is_head_enriched(self) -> None:
        from vision_spectra.metrics.gradient_alignment import head_k_for_rank, subspace_alignment

        w, u, vt = self._basis()
        k = head_k_for_rank(min(w.shape))
        g = -(u[:, :k] @ vt[:k])  # −U_k V_kᵀ: the step GROWS the top-k σ
        out = subspace_alignment(g, w)
        assert out["cos_head"] == pytest.approx(1.0, abs=1e-9)
        assert out["cos_tail"] == pytest.approx(0.0, abs=1e-9)
        assert out["head_energy"] == pytest.approx(1.0, abs=1e-9)
        # baseline k²/(mn) ⇒ enrichment = mn/k²
        assert out["head_energy_enrichment"] == pytest.approx(w.size / k**2, rel=1e-6)

    def test_isotropic_gradient_has_unit_enrichment(self) -> None:
        from vision_spectra.metrics.gradient_alignment import subspace_alignment

        rng = np.random.default_rng(1)
        w = rng.normal(size=(256, 256))
        g = rng.normal(size=(256, 256))
        out = subspace_alignment(g, w)
        assert 0.7 < out["head_energy_enrichment"] < 1.3
        assert 0.9 < out["tail_energy_enrichment"] < 1.1
        assert abs(out["cos_tail"]) < 0.1 and abs(out["cos_head"]) < 0.1

    def test_legacy_cosine_unchanged_and_qkv_split(self) -> None:
        from vision_spectra.metrics.gradient_alignment import (
            analyze_model_gradient_alignment,
            compute_gradient_alignment,
        )

        rng = np.random.default_rng(2)
        w = rng.normal(size=(40, 24))
        g = rng.normal(size=(40, 24))
        u, _, vt = np.linalg.svd(w, full_matrices=False)
        uv = u @ vt
        legacy = float(np.sum(g * uv) / (np.linalg.norm(g) * np.linalg.norm(uv)))
        assert compute_gradient_alignment(g, w).cosine_similarity == pytest.approx(legacy)

        # fused qkv is split into q/k/v entries named like the spectral extraction
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=3,
            img_size=28,
            patch_size=4,
            embed_dim=32,
            depth=1,
            num_heads=1,
        )
        out = model(torch.randn(4, 3, 28, 28))
        out.sum().backward()
        results = analyze_model_gradient_alignment(model, layer_patterns=["attn", "mlp"])
        names = {r.layer_name for r in results}
        assert {"blocks.0.attn.qkv.q", "blocks.0.attn.qkv.k", "blocks.0.attn.qkv.v"} <= names
        assert {r.matrix_type for r in results} == {"q", "k", "v", "proj", "fc1", "fc2"}
        agg = __import__(
            "vision_spectra.metrics.gradient_alignment", fromlist=["aggregate_gradient_alignment"]
        ).aggregate_gradient_alignment(results)
        assert {"cos_tail_mean", "cos_head_mean", "fc1_cos_tail", "q_cos_head"} <= set(agg)
        assert all(np.isfinite(agg[k]) for k in ("cos_tail_mean", "cos_head_mean"))


class TestHeadDropProbe:
    """Absolute-count head drop removes exactly the top-n singular values."""

    def test_drop_top_n(self) -> None:
        from vision_spectra.metrics.tail_truncation import drop_top_singular_values

        rng = np.random.RandomState(0)
        u, _ = np.linalg.qr(rng.randn(10, 5))
        v, _ = np.linalg.qr(rng.randn(5, 5))
        w = (u * np.array([10.0, 3.0, 1.0, 0.3, 0.1])) @ v.T
        new_w, info = drop_top_singular_values(w, 2)
        kept = np.sort(np.linalg.svd(new_w, compute_uv=False))[::-1][:3]
        assert np.allclose(kept, [1.0, 0.3, 0.1], atol=1e-8)
        assert info["n_dropped"] == 2 and info["top_sv_after"] == pytest.approx(1.0)
        # never drops everything
        _, info = drop_top_singular_values(w, 99)
        assert info["n_dropped"] == 4

    def test_groups_split_qkv_and_runner_logs_headn(self, tmp_path) -> None:
        import mlflow

        from vision_spectra.metrics.tail_truncation import iter_target_matrices

        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=3,
            img_size=28,
            patch_size=4,
            embed_dim=32,
            depth=2,
            num_heads=1,
        )
        types = [t for *_, t in iter_target_matrices(model, "all")]
        assert sorted(set(types)) == ["fc1", "fc2", "k", "proj", "q", "v"]
        assert len(types) == 12  # 6 matrices × 2 blocks (qkv split)
        assert [t for *_, t in iter_target_matrices(model, "mlp")] == ["fc1", "fc2"] * 2

        x = torch.randn(16, 3, 28, 28)
        y = torch.randint(0, 3, (16,))
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=8)
        mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
        with mlflow.start_run() as run:
            summary = run_truncation_analysis(
                model, loader, torch.device("cpu"), max_batches=2, head_drop_counts=(1, 3)
            )
        assert set(summary) == {"bulk", "head", "head_drop"}
        groups = {(r["group"], r["n_drop"]) for r in summary["head_drop"]["results"]}
        assert {("all", 1), ("all", 3), ("qkv", 1), ("mlp", 3)} <= groups
        client = mlflow.MlflowClient()
        steps = {
            m.step for m in client.get_metric_history(run.info.run_id, "truncation/headn_acc")
        }
        assert steps == {1, 3}
        assert client.get_metric_history(run.info.run_id, "truncation/headn_mlp_acc")


class TestFollowupStudy:
    """The follow-up cells are step-matched and carry the intended knobs."""

    def test_step_matching(self) -> None:
        from vision_spectra.experiments.run_spectral_analysis import (
            build_followup_configs,
            expected_total_steps,
        )

        cells = {c.name: c for c in build_followup_configs(wide=True)}
        long_syn = expected_total_steps(cells["w192_synlong"])
        path = expected_total_steps(cells["w192_path"])
        assert abs(long_syn - path) / path < 0.01  # ≈70k steps both
        long_1k = expected_total_steps(cells["w192_synlong1k"])
        assert abs(long_1k - path) / path < 0.01  # steps-only control, same ≈70k budget
        assert cells["w192_synlong1k"].num_samples == 1000
        assert cells["w192_synlong1k"].epochs == 2200
        # spectral snapshots land at the same step counts as the 50-epoch cells
        le = cells["w192_synlong1k"].log_epochs
        assert le[0] == 0 and le[-1] == 2154 and 220 in le and 1319 in le  # final logged by runner
        assert cells["w192_synlong1k"].val_every == 44
        assert cells["w192_synlong1k"].warmup_epochs == 220
        assert all(
            c.val_every == 1 and c.warmup_epochs == 5
            for n, c in cells.items()
            if n != "w192_synlong1k"
        )
        short_path = expected_total_steps(cells["w192_pathshort"])
        syn = expected_total_steps(cells["w192_syn"])
        assert abs(short_path - syn) / syn < 0.05  # ≈0.9k steps both
        assert cells["w192_pathshort"].train_subsample == 1000
        assert cells["w192_synlong"].num_eval_samples == 2000
        assert (
            cells["w192_path_wd0"].weight_decay == 0.0
            and cells["w384_path_wd0"].weight_decay == 0.0
        )
        assert all(c.save_checkpoint and not c.log_histograms for c in cells.values())
        assert {"w768_path", "w024_path"} <= set(cells)
        for cfg in cells.values():
            num_heads = max(1, cfg.embed_dim // 32)
            assert cfg.embed_dim % num_heads == 0
        assert len({c.name for c in build_followup_configs(wide=True)}) == len(cells)  # unique

    def test_expected_steps_match_real_loaders(self) -> None:
        import torch

        from vision_spectra.data.synthetic import create_synthetic_dataset
        from vision_spectra.experiments.run_spectral_analysis import (
            _PATH,
            _SIMPLE,
            _cell,
            expected_total_steps,
        )

        syn = _cell("w192", 192, 6, _SIMPLE)  # 1000 samples, bs 32, 30 epochs, no drop_last
        train_loader, _, _ = create_synthetic_dataset(
            num_classes=3,
            num_samples_train=1000,
            num_samples_val=8,
            num_samples_test=8,
            batch_size=32,
        )
        assert len(train_loader) * syn.epochs == expected_total_steps(syn) == 960
        # MedMNIST train loaders use drop_last=True → floor(n / bs)
        short = _cell("w192", 192, 6, _PATH)
        short.train_subsample, short.epochs, short.batch_size = 1000, 30, 32
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(1000, 1)), batch_size=32, drop_last=True
        )
        assert len(dl) * 30 == expected_total_steps(short) == 930

    def test_study_set_dispatch(self) -> None:
        from vision_spectra.experiments.run_spectral_analysis import build_study_set

        assert len(build_study_set("tiers", 1)) == 8
        assert len(build_study_set("followup")) == 13
        assert len(build_study_set("followup-wide")) == 15
        with pytest.raises(ValueError):
            build_study_set("nope")

    def test_latest_run_per_seed_dedup(self) -> None:
        import pandas as pd

        from vision_spectra.analysis.publication_figures import select_latest_run_per_seed

        runs = pd.DataFrame(
            {
                "run_id": ["a", "b", "c", "d", "e", "f"],
                "params.seed": ["42", "142", "42", "242", None, None],
                "start_time": pd.to_datetime([1, 2, 3, 4, 5, 6], unit="s"),
            }
        )
        kept = select_latest_run_per_seed(runs)
        # "a" superseded by "c"; seedless rows "e"/"f" are kept as-is (not merged).
        assert sorted(kept["run_id"]) == ["b", "c", "d", "e", "f"]
        # no seed column at all → untouched
        assert len(select_latest_run_per_seed(runs.drop(columns=["params.seed"]))) == 6


class TestSyntheticCacheAndSubsample:
    def test_cached_dataset_matches_on_the_fly(self) -> None:
        from vision_spectra.data.synthetic import SyntheticImageDataset
        from vision_spectra.data.transforms import get_eval_transforms

        t = get_eval_transforms(28, 3)
        a = SyntheticImageDataset(40, 3, seed=5, transform=t, cache=True)
        b = SyntheticImageDataset(40, 3, seed=5, transform=t, cache=False)
        assert a._cache is not None and a._cache.shape == (40, 28, 28, 3)
        for i in (0, 7, 39):
            xa, ya = a[i]
            xb, yb = b[i]
            assert ya == yb and torch.equal(xa, xb)

    def test_train_only_subsample_allocation(self) -> None:
        from vision_spectra.data.medmnist import MedMNISTDataset
        from vision_spectra.settings import DatasetConfig

        np.random.seed(0)
        labels = np.repeat(np.arange(9), [2000, 1500, 1000, 1000, 1000, 1000, 500, 500, 496])
        idx = MedMNISTDataset._get_stratified_count_indices(labels, 1000)
        assert len(idx) == 1000 and len(set(idx)) == 1000
        counts = np.bincount(labels[idx], minlength=9)
        assert counts.min() >= 1 and abs(counts[0] - 222) <= 1  # ≈ proportional
        cfg = DatasetConfig(train_subsample=1000)
        assert cfg.train_subsample == 1000
        with pytest.raises(ValidationError):
            DatasetConfig(train_subsample=0)
