#!/usr/bin/env python3
"""Generate scientifically defensible evaluation charts for CNTT-KLCN155.

Important
---------
This script DOES NOT fabricate training curves or evaluation metrics and does not
unpickle .pt checkpoints. It uses source CSV files produced by training/evaluation.
If a source file or required metric is missing, the relevant chart is skipped and
recorded in chart_generation_audit.json rather than filled with default values.

Expected CSV inputs (relative to --root unless absolute paths are supplied):
  reports/metrics/ssdlite_training_log.csv
      Required: epoch; one or more loss columns and map50 (or map50_95).
      Recommended columns: epoch, train_loss, val_loss, map50, map50_95.
  artifacts/detection_v2/E1_yolov8n_10cls/results.csv
      Standard Ultralytics training results.csv is supported.
  reports/metrics/classification_metrics.csv
      Required columns: model and at least one of accuracy, macro_f1, weighted_f1,
      precision_macro, recall_macro. Optional: val_loss and split (validation/test).
      Metrics must be fractions in [0, 1], not percentages.
  reports/metrics/per_class_ap.csv
      Required columns: model, class, ap50. Optional: ap50_95.
      AP values must be fractions in [0, 1], not percentages.
  reports/metrics/confusion_matrix.csv
      Long form: true_class, pred_class, count. Include background as a class
      when the evaluator uses it for missed/false detections.
  reports/metrics/pr_curve.csv
      Required: model, recall, precision; values in [0, 1]. One row per curve point.
      Optional: class, which creates separate curves for each model/class pair.
  reports/metrics/f1_confidence.csv
      Required: model, confidence, precision, recall; values in [0, 1]. F1 is
      computed from precision and recall, not manually entered.
  reports/metrics/ablation_benchmark.csv
      Required: config, latency_ms and at least one of precision, recall, map50,
      map50_95. Quality metrics are fractions in [0, 1]. Latency is milliseconds
      per image, measured under a documented/consistent hardware setup.

Dependencies: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Use a non-interactive backend so the script works on servers/CI without a display.
MPL_CONFIG_DIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib_cache"))
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_ORDER = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]
BACKGROUND_LABEL = "background"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.45,
})


def norm_col(value: Any) -> str:
    """Normalize a CSV column name without destroying Ultralytics slash names."""
    return str(value).strip().lower()


def resolve_column(df: pd.DataFrame, candidates: tuple[str, ...] | list[str]) -> str | None:
    columns = {norm_col(c): c for c in df.columns}
    for candidate in candidates:
        if norm_col(candidate) in columns:
            return columns[norm_col(candidate)]
    return None


def read_csv(path: Path, required: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Không đọc được CSV {path}: {exc}") from exc
    if df.empty:
        raise ValueError(f"File CSV rỗng: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric_series(df: pd.DataFrame, col: str, source: Path) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Cột '{col}' trong {source} không có giá trị số hợp lệ.")
    return values


def validate_unit_interval(series: pd.Series, name: str, source: Path) -> None:
    values = series.dropna()
    if len(values) and ((values < 0).any() or (values > 1).any()):
        raise ValueError(
            f"Cột '{name}' trong {source} phải dùng tỷ lệ [0, 1], ví dụ 0.1971, "
            "không phải phần trăm 19.71. Hãy chuẩn hóa dữ liệu nguồn trước khi vẽ."
        )


def save_figure(fig: plt.Figure, path: Path, generated: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(str(path))
    print(f"[OK] Đã lưu: {path}")


def note_skip(name: str, reason: str, skipped: list[dict[str, str]]) -> None:
    print(f"[SKIP] {name}: {reason}")
    skipped.append({"chart": name, "reason": reason})


def find_yolo_log(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit if explicit.is_absolute() else root / explicit
    expected = root / "artifacts/detection_v2/E1_yolov8n_10cls/results.csv"
    if expected.exists():
        return expected
    model_dir = root / "artifacts/detection_v2/E1_yolov8n_10cls"
    if model_dir.exists():
        matches = sorted(model_dir.glob("**/results.csv"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print("[WARN] Tìm thấy nhiều results.csv của YOLO; dùng file đầu tiên theo thứ tự đường dẫn:")
            for item in matches:
                print(f"       {item}")
            return matches[0]
    return expected


def _plot_training_log(
    df: pd.DataFrame,
    source: Path,
    model_name: str,
    output_path: Path,
    generated: list[str],
    skipped: list[dict[str, str]],
    yolo_style: bool = False,
) -> dict[str, Any] | None:
    epoch_col = resolve_column(df, ("epoch", "epochs"))
    if epoch_col is None:
        note_skip(model_name, f"{source} thiếu cột epoch; không tự suy đoán trục epoch.", skipped)
        return None
    epochs = numeric_series(df, epoch_col, source)

    fig, (loss_ax, map_ax) = plt.subplots(1, 2, figsize=(13, 5.2))
    n_loss_lines = 0
    metric_cols: dict[str, str] = {}

    if yolo_style:
        loss_candidates = [
            ("train/box_loss", ("train/box_loss",)),
            ("train/cls_loss", ("train/cls_loss",)),
            ("train/dfl_loss", ("train/dfl_loss",)),
            ("val/box_loss", ("val/box_loss",)),
            ("val/cls_loss", ("val/cls_loss",)),
            ("val/dfl_loss", ("val/dfl_loss",)),
        ]
        for label, aliases in loss_candidates:
            col = resolve_column(df, aliases)
            if col:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().any():
                    loss_ax.plot(epochs, y, label=label, linewidth=1.8)
                    n_loss_lines += 1
        map_candidates = [
            ("mAP@0.5", ("metrics/mAP50(B)", "metrics/mAP50", "map50", "val_map50")),
            ("mAP@0.5:0.95", ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "val_map50_95")),
        ]
        metric_cols = {
            "precision": resolve_column(df, ("metrics/precision(B)", "metrics/precision", "precision")) or "",
            "recall": resolve_column(df, ("metrics/recall(B)", "metrics/recall", "recall")) or "",
        }
    else:
        loss_candidates = [
            ("Train Loss", ("train_loss", "train_total_loss", "train/loss", "loss_train")),
            ("Validation Loss", ("val_loss", "validation_loss", "val_total_loss", "val/loss", "loss_val")),
        ]
        for label, aliases in loss_candidates:
            col = resolve_column(df, aliases)
            if col:
                y = pd.to_numeric(df[col], errors="coerce")
                if y.notna().any():
                    loss_ax.plot(epochs, y, label=label, linewidth=2)
                    n_loss_lines += 1
        map_candidates = [
            ("mAP@0.5", ("map50", "val_map50", "validation/map50", "metrics/map50")),
            ("mAP@0.5:0.95", ("map50_95", "val_map50_95", "validation/map50_95", "metrics/map50-95")),
        ]
        metric_cols = {
            "precision": resolve_column(df, ("precision", "val_precision")) or "",
            "recall": resolve_column(df, ("recall", "val_recall")) or "",
        }

    n_map_lines = 0
    for label, aliases in map_candidates:
        col = resolve_column(df, aliases)
        if col:
            y = pd.to_numeric(df[col], errors="coerce")
            if y.notna().any():
                # Metrics are expected as fractions, consistent with evaluation CSVs.
                if y.dropna().max() > 1.0:
                    note_skip(model_name, f"Cột {col} trong {source} có giá trị > 1; hãy xác nhận đơn vị trước khi vẽ.", skipped)
                    plt.close(fig)
                    return None
                map_ax.plot(epochs, y * 100, label=label, linewidth=2)
                n_map_lines += 1

    if n_loss_lines == 0 or n_map_lines == 0:
        plt.close(fig)
        note_skip(model_name, f"{source} cần ít nhất một cột Loss và một cột mAP thực tế để vẽ.", skipped)
        return None

    loss_ax.set_title(f"{model_name} — Loss theo epoch")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend(loc="best")
    loss_ax.grid(True)
    map_ax.set_title(f"{model_name} — mAP theo epoch")
    map_ax.set_xlabel("Epoch")
    map_ax.set_ylabel("mAP (%)")
    map_ax.set_ylim(bottom=0)
    map_ax.legend(loc="best")
    map_ax.grid(True)
    fig.suptitle(f"{model_name}: lịch sử huấn luyện từ {source.name}")
    fig.tight_layout()
    save_figure(fig, output_path, generated)

    # Report the row with highest logged mAP@0.5 if available, not a hard-coded epoch.
    map50_col = resolve_column(df, ("metrics/mAP50(B)", "metrics/mAP50", "map50", "val_map50", "validation/map50"))
    map95_col = resolve_column(df, ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95", "val_map50_95", "validation/map50_95"))
    summary: dict[str, Any] = {"model": model_name, "source_log": str(source), "epochs_logged": int(epochs.notna().sum())}
    if map50_col:
        map50 = pd.to_numeric(df[map50_col], errors="coerce")
        if map50.notna().any():
            idx = map50.idxmax()
            summary["best_logged_epoch_by_map50"] = float(pd.to_numeric(df.loc[idx, epoch_col], errors="coerce"))
            summary["best_logged_map50"] = float(map50.loc[idx])
            if map95_col:
                v = pd.to_numeric(df.loc[idx, map95_col], errors="coerce")
                summary["map50_95_at_best_map50"] = None if pd.isna(v) else float(v)
            for key, col in metric_cols.items():
                if col:
                    v = pd.to_numeric(df.loc[idx, col], errors="coerce")
                    summary[f"{key}_at_best_map50"] = None if pd.isna(v) else float(v)
    return summary


def plot_classification_metrics(path: Path, output_path: Path, generated: list[str], skipped: list[dict[str, str]]) -> None:
    """Plot classification metrics independently from object-detection AP."""
    df = read_csv(path)
    if df is None:
        note_skip("Classification metrics", f"Chưa có file dữ liệu {path}.", skipped)
        return
    model_col = resolve_column(df, ("model", "model_name", "classifier"))
    if model_col is None:
        note_skip("Classification metrics", f"{path} phải có cột model.", skipped)
        return
    metric_aliases = {
        "Accuracy": ("accuracy", "acc", "val_acc"),
        "Macro-F1": ("macro_f1", "f1_macro", "val_macro_f1"),
        "Weighted-F1": ("weighted_f1", "f1_weighted"),
        "Macro-Precision": ("precision_macro", "macro_precision"),
        "Macro-Recall": ("recall_macro", "macro_recall"),
    }
    metric_cols = {label: resolve_column(df, aliases) for label, aliases in metric_aliases.items()}
    metric_cols = {label: col for label, col in metric_cols.items() if col is not None}
    if not metric_cols:
        note_skip("Classification metrics", f"{path} thiếu các cột accuracy/macro_f1/weighted_f1/precision_macro/recall_macro.", skipped)
        return
    split_col = resolve_column(df, ("split", "evaluation_split", "partition"))
    loss_col = resolve_column(df, ("val_loss", "validation_loss", "loss"))
    data = df.copy()
    display_names = data[model_col].astype(str)
    if split_col:
        display_names = display_names + " (" + data[split_col].astype(str) + ")"
    data["_display_name"] = display_names
    usable_metric_cols: dict[str, str] = {}
    for label, col in metric_cols.items():
        data[col] = pd.to_numeric(data[col], errors="coerce")
        validate_unit_interval(data[col], col, path)
        if data[col].notna().any():
            usable_metric_cols[label] = col
    metric_cols = usable_metric_cols
    if not metric_cols:
        note_skip("Classification metrics", f"Các cột metric trong {path} không có giá trị số hợp lệ.", skipped)
        return
    if loss_col:
        data[loss_col] = pd.to_numeric(data[loss_col], errors="coerce")
        if data[loss_col].dropna().lt(0).any():
            note_skip("Classification metrics", f"Loss trong {path} có giá trị âm.", skipped)
            return
    if data["_display_name"].duplicated().any():
        note_skip("Classification metrics", f"{path} có nhiều dòng cùng model/split; hãy xác định cách tổng hợp hoặc ghi rõ seed/run.", skipped)
        return

    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 2 if loss_col and data[loss_col].notna().any() else 1,
                             figsize=(max(9, len(data) * 1.5), 5.5), squeeze=False)
    score_ax = axes[0, 0]
    width = 0.82 / len(metric_cols)
    for i, (label, col) in enumerate(metric_cols.items()):
        vals = data[col].to_numpy(dtype=float) * 100
        bars = score_ax.bar(x + (i - (len(metric_cols)-1)/2) * width, vals, width, label=label)
        for bar in bars:
            value = bar.get_height()
            if np.isfinite(value):
                score_ax.annotate(f"{value:.1f}", (bar.get_x()+bar.get_width()/2, value),
                                  xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=90)
    score_ax.set_xticks(x)
    score_ax.set_xticklabels(data["_display_name"], rotation=25, ha="right")
    score_ax.set_ylim(0, 105)
    score_ax.set_ylabel("Score (%)")
    score_ax.set_title("Chỉ số phân loại")
    score_ax.legend(loc="best")
    if axes.shape[1] == 2:
        loss_ax = axes[0, 1]
        loss_ax.bar(x, data[loss_col].to_numpy(dtype=float), width=0.62, label=loss_col)
        loss_ax.set_xticks(x)
        loss_ax.set_xticklabels(data["_display_name"], rotation=25, ha="right")
        loss_ax.set_ylabel("Loss")
        loss_ax.set_title("Validation Loss")
    fig.suptitle("Đánh giá bộ phân loại — chỉ số lấy từ CSV nguồn")
    fig.tight_layout()
    save_figure(fig, output_path, generated)


def plot_per_class_ap(path: Path, output_path: Path, generated: list[str], skipped: list[dict[str, str]]) -> None:
    df = read_csv(path)
    if df is None:
        note_skip("AP theo lớp", f"Chưa có file dữ liệu {path}.", skipped)
        return
    model_col = resolve_column(df, ("model", "model_name"))
    class_col = resolve_column(df, ("class", "class_name", "category"))
    ap_col = resolve_column(df, ("ap50", "ap@0.5", "map50"))
    if not all((model_col, class_col, ap_col)):
        note_skip("AP theo lớp", f"{path} phải có các cột model, class, ap50.", skipped)
        return
    data = df[[model_col, class_col, ap_col]].copy()
    data.columns = ["model", "class", "ap50"]
    data["ap50"] = pd.to_numeric(data["ap50"], errors="coerce")
    data = data.dropna(subset=["model", "class", "ap50"])
    if data.empty:
        note_skip("AP theo lớp", f"Không có AP hợp lệ trong {path}.", skipped)
        return
    validate_unit_interval(data["ap50"], "ap50", path)
    data["ap50"] *= 100
    if data.duplicated(["model", "class"]).any():
        note_skip("AP theo lớp", f"{path} có nhiều dòng trùng model/class; cần tổng hợp rõ ràng trước khi vẽ.", skipped)
        return
    ap95_col = resolve_column(df, ("ap50_95", "ap@0.5:0.95", "map50_95", "ap5095"))
    if ap95_col:
        data[ap95_col] = pd.to_numeric(df.loc[data.index, ap95_col], errors="coerce")
        validate_unit_interval(data[ap95_col], ap95_col, path)
        if data[ap95_col].notna().any():
            data["ap50_95"] = data[ap95_col] * 100
        else:
            ap95_col = None
    classes_present = [c for c in CLASS_ORDER if c in set(data["class"].astype(str))]
    classes_present += sorted(set(data["class"].astype(str)) - set(classes_present) - {BACKGROUND_LABEL})
    models = list(dict.fromkeys(data["model"].astype(str)))
    fig, axes = plt.subplots(1, 2 if ap95_col else 1,
                             figsize=(max(11, len(classes_present) * (2.0 if ap95_col else 1.1)), 6), squeeze=False)
    metric_sets = [("ap50", "AP@0.5 (%)", axes[0, 0])]
    if ap95_col:
        metric_sets.append(("ap50_95", "AP@0.5:0.95 (%)", axes[0, 1]))
    for metric_key, ylabel, ax in metric_sets:
        x = np.arange(len(classes_present))
        width = 0.82 / max(1, len(models))
        for i, model in enumerate(models):
            subset = data[data["model"].astype(str) == model].set_index("class")
            vals = [subset.at[c, metric_key] if c in subset.index and pd.notna(subset.at[c, metric_key]) else np.nan
                    for c in classes_present]
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=model)
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h):
                    ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h), xytext=(0, 2),
                                textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(classes_present, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.set_title(f"So sánh {ylabel} theo lớp")
        ax.legend(loc="best")
    fig.suptitle("AP theo lớp — chỉ số lấy từ báo cáo đánh giá")
    fig.tight_layout()
    save_figure(fig, output_path, generated)


def plot_confusion_matrix(path: Path, output_path: Path, generated: list[str], skipped: list[dict[str, str]]) -> None:
    df = read_csv(path)
    if df is None:
        note_skip("Confusion matrix", f"Chưa có file dữ liệu {path}.", skipped)
        return
    true_col = resolve_column(df, ("true_class", "ground_truth", "gt_class"))
    pred_col = resolve_column(df, ("pred_class", "predicted_class", "prediction"))
    count_col = resolve_column(df, ("count", "n", "samples"))
    if not all((true_col, pred_col, count_col)):
        note_skip("Confusion matrix", f"{path} phải có các cột true_class, pred_class, count.", skipped)
        return
    data = df[[true_col, pred_col, count_col]].copy()
    data.columns = ["true_class", "pred_class", "count"]
    data["count"] = pd.to_numeric(data["count"], errors="coerce")
    if data["count"].isna().any() or (data["count"] < 0).any():
        note_skip("Confusion matrix", f"Cột count trong {path} có giá trị thiếu/âm/không hợp lệ.", skipped)
        return
    if not np.allclose(data["count"], np.round(data["count"])):
        note_skip("Confusion matrix", f"Cột count trong {path} phải là số đếm nguyên.", skipped)
        return
    if data["count"].sum() <= 0:
        note_skip("Confusion matrix", f"Tổng số đếm trong {path} bằng 0; không thể đánh giá từ ma trận rỗng.", skipped)
        return
    labels_seen = set(data["true_class"].astype(str)) | set(data["pred_class"].astype(str))
    labels = [c for c in CLASS_ORDER if c in labels_seen]
    labels += sorted(labels_seen - set(labels) - {BACKGROUND_LABEL})
    if BACKGROUND_LABEL in labels_seen:
        labels.append(BACKGROUND_LABEL)
    if not labels:
        note_skip("Confusion matrix", f"Không có nhãn trong {path}.", skipped)
        return
    matrix = pd.pivot_table(data, index="true_class", columns="pred_class", values="count", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    counts = matrix.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, row_sums, out=np.full_like(counts, np.nan), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    masked = np.ma.masked_invalid(normalized)
    im = axes[0].imshow(masked, vmin=0, vmax=1, cmap="Blues")
    axes[0].set_title("Confusion matrix chuẩn hóa theo nhãn thực")
    axes[0].set_xlabel("Nhãn dự đoán")
    axes[0].set_ylabel("Nhãn thực tế")
    axes[1].imshow(counts, cmap="Blues")
    axes[1].set_title("Confusion matrix — số đếm")
    axes[1].set_xlabel("Nhãn dự đoán")
    axes[1].set_ylabel("Nhãn thực tế")
    for ax in axes:
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(normalized[i, j]) and (normalized[i, j] >= 0.04 or i == j):
                axes[0].text(j, i, f"{normalized[i,j]:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if normalized[i,j] > 0.5 else "black")
            if counts[i, j] > 0:
                axes[1].text(j, i, f"{int(counts[i,j])}", ha="center", va="center", fontsize=7,
                             color="white" if counts[i,j] > counts.max()/2 else "black")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Tỷ lệ theo hàng")
    fig.suptitle("Ma trận nhầm lẫn từ số liệu đánh giá; cách ghép detection phải được xác định ở bước evaluator")
    fig.tight_layout()
    save_figure(fig, output_path, generated)


def plot_pr_and_f1(pr_path: Path, f1_path: Path, output_path: Path, generated: list[str], skipped: list[dict[str, str]]) -> None:
    pr = read_csv(pr_path)
    f1_data = read_csv(f1_path)
    if pr is None or f1_data is None:
        missing = [str(p) for p, d in ((pr_path, pr), (f1_path, f1_data)) if d is None]
        note_skip("PR/F1", "Thiếu file nguồn: " + ", ".join(missing), skipped)
        return
    pr_model = resolve_column(pr, ("model", "model_name"))
    recall_col = resolve_column(pr, ("recall",))
    precision_col = resolve_column(pr, ("precision",))
    f_model = resolve_column(f1_data, ("model", "model_name"))
    conf_col = resolve_column(f1_data, ("confidence", "conf", "threshold"))
    f_precision = resolve_column(f1_data, ("precision",))
    f_recall = resolve_column(f1_data, ("recall",))
    if not all((pr_model, recall_col, precision_col)):
        note_skip("PR/F1", f"{pr_path} phải có model, recall, precision.", skipped)
        return
    if not all((f_model, conf_col, f_precision, f_recall)):
        note_skip("PR/F1", f"{f1_path} phải có model, confidence, precision, recall.", skipped)
        return

    pr[recall_col] = pd.to_numeric(pr[recall_col], errors="coerce")
    pr[precision_col] = pd.to_numeric(pr[precision_col], errors="coerce")
    f1_data[conf_col] = pd.to_numeric(f1_data[conf_col], errors="coerce")
    f1_data[f_precision] = pd.to_numeric(f1_data[f_precision], errors="coerce")
    f1_data[f_recall] = pd.to_numeric(f1_data[f_recall], errors="coerce")
    for col, name in ((recall_col, "recall"), (precision_col, "precision")):
        validate_unit_interval(pr[col], name, pr_path)
    for col, name in ((f_precision, "precision"), (f_recall, "recall"), (conf_col, "confidence")):
        validate_unit_interval(f1_data[col], name, f1_path)
    pr = pr.dropna(subset=[pr_model, recall_col, precision_col])
    f1_data = f1_data.dropna(subset=[f_model, conf_col, f_precision, f_recall])
    if pr.empty or f1_data.empty:
        note_skip("PR/F1", "Không có dòng dữ liệu hợp lệ sau khi loại giá trị thiếu.", skipped)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    pr_class_col = resolve_column(pr, ("class", "class_name"))
    for keys, group in pr.groupby([pr_model, pr_class_col] if pr_class_col else [pr_model], dropna=False):
        group = group.sort_values(recall_col)
        if pr_class_col:
            model, cls = keys
            label = f"{model} / {cls}"
        else:
            label = str(keys)
        ax1.plot(group[recall_col], group[precision_col], label=label, linewidth=1.8)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_title("Precision–Recall curve")
    ax1.legend(loc="best", fontsize=7.5)

    for model, group in f1_data.groupby(f_model):
        group = group.sort_values(conf_col).copy()
        denom = group[f_precision] + group[f_recall]
        group["f1_computed"] = np.divide(
            2 * group[f_precision] * group[f_recall], denom,
            out=np.zeros(len(group), dtype=float), where=denom.to_numpy() > 0,
        )
        best_idx = group["f1_computed"].idxmax()
        best = group.loc[best_idx]
        ax2.plot(group[conf_col], group["f1_computed"], label=str(model), linewidth=1.8)
        ax2.scatter([best[conf_col]], [best["f1_computed"]], s=20)
        ax2.annotate(f"F1={best['f1_computed']:.3f}\nconf={best[conf_col]:.2f}",
                     (best[conf_col], best["f1_computed"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax2.set_xlabel("Confidence threshold")
    ax2.set_ylabel("F1")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("F1 theo confidence (tính từ Precision/Recall)")
    ax2.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_path, generated)


def plot_ablation(path: Path, output_path: Path, generated: list[str], skipped: list[dict[str, str]]) -> None:
    df = read_csv(path)
    if df is None:
        note_skip("Ablation/benchmark", f"Chưa có file dữ liệu {path}.", skipped)
        return
    config_col = resolve_column(df, ("config", "configuration", "model", "variant"))
    latency_col = resolve_column(df, ("latency_ms", "latency", "ms_per_image"))
    if config_col is None or latency_col is None:
        note_skip("Ablation/benchmark", f"{path} phải có config và latency_ms.", skipped)
        return
    metric_aliases = {
        "Precision": ("precision",),
        "Recall": ("recall",),
        "mAP@0.5": ("map50", "ap50"),
        "mAP@0.5:0.95": ("map50_95", "map5095", "ap50_95"),
    }
    metric_cols = {label: resolve_column(df, aliases) for label, aliases in metric_aliases.items()}
    metric_cols = {k: v for k, v in metric_cols.items() if v is not None}
    if not metric_cols:
        note_skip("Ablation/benchmark", f"{path} thiếu precision, recall, map50 và map50_95.", skipped)
        return
    data = df.copy()
    data[latency_col] = pd.to_numeric(data[latency_col], errors="coerce")
    if data[latency_col].isna().any() or (data[latency_col] <= 0).any():
        note_skip("Ablation/benchmark", f"Latency trong {path} phải là số dương, không thiếu.", skipped)
        return
    usable_metric_cols: dict[str, str] = {}
    for label, col in metric_cols.items():
        data[col] = pd.to_numeric(data[col], errors="coerce")
        validate_unit_interval(data[col], col, path)
        if data[col].notna().any():
            usable_metric_cols[label] = col
    metric_cols = usable_metric_cols
    if not metric_cols:
        note_skip("Ablation/benchmark", f"Các cột metric trong {path} không có giá trị số hợp lệ.", skipped)
        return
    if data[config_col].duplicated().any():
        note_skip("Ablation/benchmark", f"{path} có tên config trùng nhau; hãy tổng hợp các lần lặp trước khi vẽ.", skipped)
        return

    configs = data[config_col].astype(str).tolist()
    x = np.arange(len(configs))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, len(configs) * 1.5), 6))
    labels = list(metric_cols)
    width = 0.82 / len(labels)
    for i, label in enumerate(labels):
        values = data[metric_cols[label]].to_numpy(dtype=float) * 100
        bars = ax1.bar(x + (i - (len(labels)-1)/2) * width, values, width, label=label)
        for bar in bars:
            value = bar.get_height()
            if np.isfinite(value):
                ax1.annotate(f"{value:.1f}", (bar.get_x()+bar.get_width()/2, value),
                             xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7, rotation=90)
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=25, ha="right")
    ax1.set_ylim(bottom=0, top=100)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Chất lượng phát hiện")
    ax1.legend(loc="best")

    lat = data[latency_col].to_numpy(dtype=float)
    bars = ax2.bar(x, lat, width=0.62, label="Latency (ms/image)")
    fps_derived = 1000.0 / lat
    for i, (bar, ms, fps) in enumerate(zip(bars, lat, fps_derived)):
        ax2.annotate(f"{ms:.1f} ms\n~{fps:.1f} FPS*", (bar.get_x()+bar.get_width()/2, ms),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=25, ha="right")
    ax2.set_ylabel("Latency (ms/image)")
    ax2.set_title("Độ trễ xử lý")
    ax2.text(0.02, 0.98, "*FPS suy ra từ 1000/latency, không phải FPS đo trực tiếp.",
             transform=ax2.transAxes, va="top", ha="left", fontsize=8)
    fig.suptitle("So sánh ablation/benchmark — các cấu hình cần đo trong cùng điều kiện")
    fig.tight_layout()
    save_figure(fig, output_path, generated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo biểu đồ báo cáo từ log/kết quả CSV có thật.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Thư mục gốc repository (mặc định: thư mục hiện tại).")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"), help="Thư mục lưu PNG.")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"), help="Thư mục CSV đánh giá chuẩn hóa.")
    parser.add_argument("--ssdlite-log", type=Path, default=None, help="CSV lịch sử train SSDLite; mặc định reports/metrics/ssdlite_training_log.csv.")
    parser.add_argument("--yolo-log", type=Path, default=None, help="Ultralytics results.csv; tự tìm trong thư mục YOLO nếu bỏ qua.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    metrics_dir = args.metrics_dir if args.metrics_dir.is_absolute() else root / args.metrics_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ssdlite_log = args.ssdlite_log or Path("reports/metrics/ssdlite_training_log.csv")
    ssdlite_log = ssdlite_log if ssdlite_log.is_absolute() else root / ssdlite_log
    yolo_log = find_yolo_log(root, args.yolo_log)

    generated: list[str] = []
    skipped: list[dict[str, str]] = []
    summaries: list[dict[str, Any]] = []
    inputs: dict[str, str | None] = {}

    # Chart 1: SSDLite learning curves from epoch-by-epoch logs.
    try:
        df = read_csv(ssdlite_log)
        inputs["ssdlite_training_log"] = str(ssdlite_log) if df is not None else None
        if df is None:
            note_skip("SSDLite Loss/mAP", f"Chưa có log lịch sử huấn luyện {ssdlite_log}; checkpoint tốt nhất không đủ để dựng đường cong theo epoch.", skipped)
        else:
            summary = _plot_training_log(df, ssdlite_log, "SSDLite320-MobileNetV3",
                                         output_dir / "ssdlite320_loss_map_curve.png", generated, skipped, False)
            if summary:
                summaries.append(summary)
    except Exception as exc:
        note_skip("SSDLite Loss/mAP", str(exc), skipped)

    # Chart 2: YOLO learning curves from Ultralytics results.csv.
    try:
        df = read_csv(yolo_log)
        inputs["yolo_training_log"] = str(yolo_log) if df is not None else None
        if df is None:
            note_skip("YOLO Loss/mAP", f"Chưa tìm thấy results.csv tại {yolo_log}; không dựng đường cong giả.", skipped)
        else:
            summary = _plot_training_log(df, yolo_log, "YOLOv8n",
                                         output_dir / "yolov8n_loss_map_curve.png", generated, skipped, True)
            if summary:
                summaries.append(summary)
    except Exception as exc:
        note_skip("YOLO Loss/mAP", str(exc), skipped)

    # Chart 3: classifier classification metrics remain separate from detector AP.
    classifier_path = metrics_dir / "classification_metrics.csv"
    inputs["classification_metrics"] = str(classifier_path) if classifier_path.exists() else None
    try:
        plot_classification_metrics(classifier_path, output_dir / "classifier_metrics_comparison.png", generated, skipped)
    except Exception as exc:
        note_skip("Classification metrics", str(exc), skipped)

    # Chart 4: AP by class from an evaluation export, not values hard-coded into Python.
    per_class_path = metrics_dir / "per_class_ap.csv"
    inputs["per_class_ap"] = str(per_class_path) if per_class_path.exists() else None
    try:
        plot_per_class_ap(per_class_path, output_dir / "per_class_ap50_comparison.png", generated, skipped)
    except Exception as exc:
        note_skip("AP theo lớp", str(exc), skipped)

    # Chart 4: confusion matrix counts exported by the evaluator.
    cm_path = metrics_dir / "confusion_matrix.csv"
    inputs["confusion_matrix"] = str(cm_path) if cm_path.exists() else None
    try:
        plot_confusion_matrix(cm_path, output_dir / "normalized_confusion_matrix.png", generated, skipped)
    except Exception as exc:
        note_skip("Confusion matrix", str(exc), skipped)

    # Chart 5: PR and F1-confidence use separate actual evaluator exports.
    pr_path = metrics_dir / "pr_curve.csv"
    f1_path = metrics_dir / "f1_confidence.csv"
    inputs["pr_curve"] = str(pr_path) if pr_path.exists() else None
    inputs["f1_confidence"] = str(f1_path) if f1_path.exists() else None
    try:
        plot_pr_and_f1(pr_path, f1_path, output_dir / "pr_and_f1_curves.png", generated, skipped)
    except Exception as exc:
        note_skip("PR/F1", str(exc), skipped)

    # Chart 6: measured ablation/benchmark summary.
    ablation_path = metrics_dir / "ablation_benchmark.csv"
    inputs["ablation_benchmark"] = str(ablation_path) if ablation_path.exists() else None
    try:
        plot_ablation(ablation_path, output_dir / "ablation_study_chart.png", generated, skipped)
    except Exception as exc:
        note_skip("Ablation/benchmark", str(exc), skipped)

    summary_path = output_dir / "training_metric_summary.csv"
    if summaries:
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        generated.append(str(summary_path))
        print(f"[OK] Đã lưu tóm tắt metrics lịch sử: {summary_path}")
    else:
        print("[INFO] Không tạo training_metric_summary.csv vì chưa có log huấn luyện hợp lệ.")

    audit = {
        "note": "This audit records only files and charts processed by this script. It does not certify the scientific validity of upstream labels/predictions.",
        "repository_root": str(root),
        "output_dir": str(output_dir),
        "inputs": inputs,
        "generated_files": generated,
        "skipped_charts": skipped,
    }
    audit_path = output_dir / "chart_generation_audit.json"
    generated.append(str(audit_path))
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Nhật ký nguồn dữ liệu: {audit_path}")
    print(f"Hoàn tất: {len(generated)} file được tạo; {len(skipped)} biểu đồ bị bỏ qua do thiếu/không hợp lệ dữ liệu.")
    if skipped:
        print("Các biểu đồ bị bỏ qua không được thay bằng dữ liệu giả hoặc giá trị mặc định.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Đã hủy bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)
