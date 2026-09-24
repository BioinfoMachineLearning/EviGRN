"""Turn evaluation JSON into metric tables and curve series.

The GUI should show AUROC, AUPRC, early precision, and ROC/PR curves when
those fields exist. It should not render the raw report object.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _ratio_label(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return str(value).strip()
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def _ratio_sort_key(label: str) -> tuple[int, float | str]:
    number = _as_float(label)
    if number is None:
        return (1, label)
    return (0, number)


def _pick_auprc(metrics: dict[str, Any]) -> float | None:
    for key in ("auprc", "auprc_macro", "aucpr_macro", "auprc_micro", "aucpr_micro"):
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _pick_auroc(metrics: dict[str, Any]) -> float | None:
    for key in ("auroc", "auroc_macro", "auroc_micro"):
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    return None


def _early_precision(metrics: dict[str, Any], k: int) -> float | None:
    for key in (f"ep_at_{k}", f"EP@{k}", f"precision_at_{k}"):
        value = _as_float(metrics.get(key))
        if value is not None:
            return value
    pk = metrics.get("precision_at_k")
    if isinstance(pk, dict):
        for key in (f"present@{k}", str(k), f"@{k}", f"ep@{k}"):
            value = _as_float(pk.get(key))
            if value is not None:
                return value
    return None


def iter_metric_blocks(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return (ratio label, metrics dict) blocks from a report.

    Supports ``results_by_ratio`` and a flat report that already stores AUROC.
    """
    by_ratio = payload.get("results_by_ratio")
    if isinstance(by_ratio, dict) and by_ratio:
        blocks: list[tuple[str, dict[str, Any]]] = []
        for raw_ratio, metrics in by_ratio.items():
            if not isinstance(metrics, dict):
                continue
            label = _ratio_label(metrics.get("negative_ratio", raw_ratio))
            blocks.append((label, metrics))
        blocks.sort(key=lambda item: _ratio_sort_key(item[0]))
        return blocks
    if _pick_auroc(payload) is not None or _pick_auprc(payload) is not None:
        label = _ratio_label(payload.get("negative_ratio", "all"))
        return [(label or "all", payload)]
    return []


def metrics_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Table of AUROC, AUPRC, and early precision (EP@k) by negative ratio."""
    rows: list[dict[str, Any]] = []
    for ratio, metrics in iter_metric_blocks(payload):
        n_matched = _as_float(metrics.get("n_matched"))
        n_pos = _as_float(metrics.get("n_positive"))
        n_neg = _as_float(metrics.get("n_negative"))
        rows.append(
            {
                "negative_ratio": ratio,
                "ratio_label": f"1:{ratio}",
                "AUROC": _pick_auroc(metrics),
                "AUPRC": _pick_auprc(metrics),
                "EP@10": _early_precision(metrics, 10),
                "EP@50": _early_precision(metrics, 50),
                "EP@100": _early_precision(metrics, 100),
                "n_matched": None if n_matched is None else int(n_matched),
                "n_positive": None if n_pos is None else int(n_pos),
                "n_negative": None if n_neg is None else int(n_neg),
            }
        )
    return pd.DataFrame(rows)


def metric_warnings(payload: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for ratio, metrics in iter_metric_blocks(payload):
        warning = metrics.get("metric_warning") or metrics.get("error")
        if warning:
            notes.append(f"1:{ratio}: {warning}")
    return notes


def _curve_xy(
    curve: dict[str, Any],
    *,
    x_keys: tuple[str, ...],
    y_keys: tuple[str, ...],
    y_std_keys: tuple[str, ...] = (),
) -> tuple[list[float], list[float], list[float] | None] | None:
    x_raw = None
    for key in x_keys:
        if key in curve:
            x_raw = curve.get(key)
            break
    y_raw = None
    for key in y_keys:
        if key in curve:
            y_raw = curve.get(key)
            break
    if not isinstance(x_raw, list) or not isinstance(y_raw, list):
        return None
    if len(x_raw) == 0 or len(x_raw) != len(y_raw):
        return None
    xs: list[float] = []
    ys: list[float] = []
    for x_val, y_val in zip(x_raw, y_raw):
        x_num = _as_float(x_val)
        y_num = _as_float(y_val)
        if x_num is None or y_num is None:
            return None
        xs.append(x_num)
        ys.append(y_num)
    std: list[float] | None = None
    for key in y_std_keys:
        raw_std = curve.get(key)
        if isinstance(raw_std, list) and len(raw_std) == len(ys):
            parsed = [_as_float(v) for v in raw_std]
            if all(v is not None for v in parsed):
                std = [float(v) for v in parsed if v is not None]
            break
    return xs, ys, std


def pr_series(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """PR curves stored on each ratio block, if present."""
    series: list[dict[str, Any]] = []
    for ratio, metrics in iter_metric_blocks(payload):
        curve = metrics.get("pr_curve")
        if not isinstance(curve, dict):
            continue
        parsed = _curve_xy(
            curve,
            x_keys=("recall",),
            y_keys=("precision_mean", "precision"),
            y_std_keys=("precision_std",),
        )
        if parsed is None:
            continue
        recall, precision, std = parsed
        prevalence = _as_float(curve.get("positive_prevalence_mean"))
        if prevalence is None:
            prevalence = _as_float(curve.get("positive_prevalence"))
        series.append(
            {
                "ratio": ratio,
                "label": f"1:{ratio}",
                "recall": recall,
                "precision": precision,
                "precision_std": std,
                "prevalence": prevalence,
            }
        )
    return series


def roc_series(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """ROC curves only when fpr/tpr arrays are stored on the report."""
    series: list[dict[str, Any]] = []
    for ratio, metrics in iter_metric_blocks(payload):
        curve = metrics.get("roc_curve")
        if not isinstance(curve, dict):
            curve = {}
            fpr = metrics.get("fpr")
            tpr = metrics.get("tpr")
            if isinstance(fpr, list) and isinstance(tpr, list):
                curve = {"fpr": fpr, "tpr": tpr}
        if not curve:
            continue
        parsed = _curve_xy(
            curve,
            x_keys=("fpr", "fpr_mean"),
            y_keys=("tpr_mean", "tpr"),
            y_std_keys=("tpr_std",),
        )
        if parsed is None:
            continue
        fpr, tpr, std = parsed
        series.append(
            {
                "ratio": ratio,
                "label": f"1:{ratio}",
                "fpr": fpr,
                "tpr": tpr,
                "tpr_std": std,
            }
        )
    return series
