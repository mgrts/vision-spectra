"""
Statistical analysis functions for comparing spectral metrics across scenarios.

This module provides:
- Hypothesis tests (t-test, Wilcoxon signed-rank)
- Effect size calculations (Cohen's d)
- Confidence intervals
- Summary statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two groups."""

    group1_name: str
    group2_name: str
    metric_name: str
    group1_mean: float
    group2_mean: float
    group1_std: float
    group2_std: float
    mean_diff: float
    t_statistic: float
    p_value_ttest: float
    p_value_wilcoxon: float | None
    cohens_d: float
    ci_lower: float
    ci_upper: float
    is_significant: bool


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group2) - np.mean(group1)) / pooled_std)


def compare_groups(
    group1: np.ndarray,
    group2: np.ndarray,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    metric_name: str = "metric",
    paired: bool = False,
) -> ComparisonResult:
    """Perform comprehensive statistical comparison between two groups."""
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    group1 = group1[np.isfinite(group1)]
    group2 = group2[np.isfinite(group2)]

    if len(group1) < 2 or len(group2) < 2:
        return ComparisonResult(
            group1_name=group1_name,
            group2_name=group2_name,
            metric_name=metric_name,
            group1_mean=np.nan,
            group2_mean=np.nan,
            group1_std=np.nan,
            group2_std=np.nan,
            mean_diff=np.nan,
            t_statistic=np.nan,
            p_value_ttest=np.nan,
            p_value_wilcoxon=None,
            cohens_d=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            is_significant=False,
        )

    m1, m2 = float(np.mean(group1)), float(np.mean(group2))
    s1, s2 = float(np.std(group1, ddof=1)), float(np.std(group2, ddof=1))
    diff = m2 - m1

    if paired and len(group1) == len(group2):
        t_stat, p_ttest = stats.ttest_rel(group1, group2)
    else:
        t_stat, p_ttest = stats.ttest_ind(group1, group2)

    p_wilcoxon = None
    try:
        if paired and len(group1) == len(group2):
            _, p_wilcoxon = stats.wilcoxon(group1, group2)
        else:
            _, p_wilcoxon = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    except Exception:
        pass

    d = cohens_d(group1, group2)

    se_diff = np.sqrt(s1**2 / len(group1) + s2**2 / len(group2))
    t_crit = stats.t.ppf(0.975, df=len(group1) + len(group2) - 2)
    ci_lower = diff - t_crit * se_diff
    ci_upper = diff + t_crit * se_diff

    return ComparisonResult(
        group1_name=group1_name,
        group2_name=group2_name,
        metric_name=metric_name,
        group1_mean=m1,
        group2_mean=m2,
        group1_std=s1,
        group2_std=s2,
        mean_diff=diff,
        t_statistic=float(t_stat),
        p_value_ttest=float(p_ttest),
        p_value_wilcoxon=float(p_wilcoxon) if p_wilcoxon is not None else None,
        cohens_d=d,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        is_significant=p_ttest < 0.05,
    )


def compare_scenarios(
    scenario_results: dict[str, dict[str, list[float]]],
    metrics: list[str],
) -> dict[tuple[str, str, str], ComparisonResult]:
    """Compare all pairs of scenarios across multiple metrics."""
    results = {}
    scenario_names = list(scenario_results.keys())

    for i, s1 in enumerate(scenario_names):
        for s2 in scenario_names[i + 1 :]:
            for metric in metrics:
                values1 = scenario_results[s1].get(metric, [])
                values2 = scenario_results[s2].get(metric, [])
                if values1 and values2:
                    result = compare_groups(
                        np.array(values1),
                        np.array(values2),
                        group1_name=s1,
                        group2_name=s2,
                        metric_name=metric,
                        paired=True,
                    )
                    results[(s1, s2, metric)] = result
    return results


def format_comparison_table(
    comparisons: dict[tuple[str, str, str], ComparisonResult],
) -> str:
    """Format comparison results as a markdown table."""
    lines = [
        "| Comparison | Metric | Mean Diff | 95% CI | Cohen's d | p-value | Sig? |",
        "|------------|--------|-----------|--------|-----------|---------|------|",
    ]
    for (s1, s2, metric), result in sorted(comparisons.items()):
        sig = "✓" if result.is_significant else "✗"
        ci = f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}]"
        lines.append(
            f"| {s1} vs {s2} | {metric} | {result.mean_diff:.3f} | {ci} | "
            f"{result.cohens_d:.2f} | {result.p_value_ttest:.4f} | {sig} |"
        )
    return "\n".join(lines)


def summarize_results(
    scenario_results: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute summary statistics for each scenario and metric."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for scenario, metrics in scenario_results.items():
        summary[scenario] = {}
        for metric, values in metrics.items():
            arr = np.array(values)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                summary[scenario][metric] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "n": float(len(arr)),
                }
            else:
                summary[scenario][metric] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "n": 0.0,
                }
    return summary


def validate_hypothesis(
    scenario_a: dict[str, list[float]],
    scenario_b: dict[str, list[float]],
    scenario_c: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Validate the three-scenario hypothesis."""
    results: dict[str, Any] = {"hypothesis_supported": True, "predictions": {}, "comparisons": {}}

    if "alpha_exponent_mean" in scenario_a and "alpha_exponent_mean" in scenario_b:
        comp = compare_groups(
            np.array(scenario_a["alpha_exponent_mean"]),
            np.array(scenario_b["alpha_exponent_mean"]),
            "A",
            "B",
            "alpha_exponent_mean",
        )
        results["comparisons"]["A_vs_B_alpha"] = comp
        prediction_met = comp.mean_diff > 0
        results["predictions"]["B_alpha_higher_than_A"] = {
            "met": prediction_met,
            "difference": comp.mean_diff,
            "p_value": comp.p_value_ttest,
            "significant": comp.is_significant,
        }
        if not prediction_met:
            results["hypothesis_supported"] = False

    if "stable_rank_mean" in scenario_a and "stable_rank_mean" in scenario_b:
        comp = compare_groups(
            np.array(scenario_a["stable_rank_mean"]),
            np.array(scenario_b["stable_rank_mean"]),
            "A",
            "B",
            "stable_rank_mean",
        )
        results["comparisons"]["A_vs_B_stable_rank"] = comp
        prediction_met = comp.mean_diff < 0
        results["predictions"]["B_stable_rank_lower_than_A"] = {
            "met": prediction_met,
            "difference": comp.mean_diff,
            "p_value": comp.p_value_ttest,
            "significant": comp.is_significant,
        }
        if not prediction_met:
            results["hypothesis_supported"] = False

    if (
        scenario_c is not None
        and "alpha_exponent_mean" in scenario_b
        and "alpha_exponent_mean" in scenario_c
    ):
        comp = compare_groups(
            np.array(scenario_b["alpha_exponent_mean"]),
            np.array(scenario_c["alpha_exponent_mean"]),
            "B",
            "C",
            "alpha_exponent_mean",
        )
        results["comparisons"]["B_vs_C_alpha"] = comp
        prediction_met = comp.mean_diff < 0
        results["predictions"]["C_alpha_lower_than_B"] = {
            "met": prediction_met,
            "difference": comp.mean_diff,
            "p_value": comp.p_value_ttest,
            "significant": comp.is_significant,
        }
        if not prediction_met:
            results["hypothesis_supported"] = False

    return results
