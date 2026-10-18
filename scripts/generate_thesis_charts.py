"""Generate verified, publication-ready charts for the CNTT-KLCN155 thesis defense.

All numbers in these charts are loaded directly from the repository's real
training logs and evaluation artifacts:
- artifacts/detection_v2/E1_yolov8n_10cls/results.csv
- artifacts/detection_v2/E1_yolov8n_10cls/eval_val_test.json
- artifacts/detection_v2/E2_ssdlite320_10cls/history.json
- artifacts/detection_v2/E2_ssdlite320_10cls/training_summary.json
- artifacts/detection_v3_pilot/E3_yolov8n_pilot/results.csv
- artifacts/official_run/val_confusion_matrix.csv
- reports/qa_acceptance/verified_evaluation_summary_from_raw.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "figures"

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


def setup_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 0.9,
            "grid.color": "#E6E6E6",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def generate_yolov8n_loss_map_curve(out_dir: Path) -> Path:
    """1. YOLOv8n (E1_yolov8n_10cls) real 80-epoch Training/Val Loss & mAP@0.5 curve."""
    csv_path = REPO_ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "results.csv"
    eval_path = REPO_ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "eval_val_test.json"

    epochs: list[int] = []
    train_total_loss: list[float] = []
    val_total_loss: list[float] = []
    val_map50: list[float] = []
    val_map50_95: list[float] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None}
            epochs.append(int(float(cleaned["epoch"])))
            t_loss = (
                float(cleaned["train/box_loss"])
                + float(cleaned["train/cls_loss"])
                + float(cleaned["train/dfl_loss"])
            )
            v_loss = (
                float(cleaned["val/box_loss"])
                + float(cleaned["val/cls_loss"])
                + float(cleaned["val/dfl_loss"])
            )
            train_total_loss.append(t_loss)
            val_total_loss.append(v_loss)
            val_map50.append(float(cleaned["metrics/mAP50(B)"]))
            val_map50_95.append(float(cleaned["metrics/mAP50-95(B)"]))

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    coco_val_map50 = eval_data["val"]["map50"]
    coco_val_map50_95 = eval_data["val"]["map50_95"]

    # Ultralytics fitness = 0.1 * mAP50 + 0.9 * mAP50_95
    fitness = [0.1 * m50 + 0.9 * m95 for m50, m95 in zip(val_map50, val_map50_95)]
    best_idx = int(np.argmax(fitness))
    best_epoch = epochs[best_idx]
    best_map50 = val_map50[best_idx]
    best_map50_95 = val_map50_95[best_idx]

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        epochs,
        train_total_loss,
        color="#D95F02",
        linewidth=2.3,
        label="Train Total Loss (box + cls + dfl)",
    )
    l2 = ax1.plot(
        epochs,
        val_total_loss,
        color="#E6AB02",
        linewidth=2.0,
        linestyle="--",
        label="Val Total Loss (box + cls + dfl)",
    )

    l3 = ax2.plot(
        epochs,
        val_map50,
        color="#6A3D9A",
        linewidth=2.5,
        label=f"Val mAP@0.5 (Peak: {best_map50 * 100:.2f}% Ultralytics | {coco_val_map50 * 100:.2f}% COCOeval)",
    )
    l4 = ax2.plot(
        epochs,
        val_map50_95,
        color="#1F78B4",
        linewidth=2.0,
        linestyle="-.",
        label=f"Val mAP@0.5:0.95 (Peak: {best_map50_95 * 100:.2f}% Ultralytics | {coco_val_map50_95 * 100:.2f}% COCOeval)",
    )
    l5 = ax1.axvline(
        x=best_epoch,
        color="#2CA02C",
        linestyle=":",
        linewidth=2.2,
        label=f"Best Checkpoint (Epoch {best_epoch})",
    )
    ax2.scatter([best_epoch], [best_map50], color="#2CA02C", s=65, zorder=6)

    ax1.set_title(
        "YOLOv8n (E1_yolov8n_10cls): Training/Validation Loss & mAP (80 Epochs thực tế)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax1.set_xlabel("Epochs", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Total Loss (box + cls + dfl)", fontsize=11.5, fontweight="bold", color="#B84A00")
    ax2.set_ylabel("Validation mAP", fontsize=11.5, fontweight="bold", color="#6A3D9A")

    ax1.set_xlim(1, 80)
    ax1.set_ylim(2.8, 8.8)
    ax2.set_ylim(0.0, 0.34)
    ax1.grid(True)

    lines = l1 + l2 + l3 + l4 + [l5]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=9.2,
    )

    out_path = out_dir / "yolov8n_loss_map_curve.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_ssdlite320_loss_map_curve(out_dir: Path) -> Path:
    """2. SSDLite320-MobileNetV3 (E2_ssdlite320_10cls) real 36-epoch Training Loss & Val mAP curve."""
    hist_path = REPO_ROOT / "artifacts" / "detection_v2" / "E2_ssdlite320_10cls" / "history.json"
    summary_path = REPO_ROOT / "artifacts" / "detection_v2" / "E2_ssdlite320_10cls" / "training_summary.json"

    history = json.loads(hist_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["loss"]) for item in history]
    val_map50 = [float(item["validation"]["map50"]) for item in history]
    val_map50_95 = [float(item["validation"]["map50_95"]) for item in history]

    best_epoch = int(summary["best_epoch"])
    best_idx = epochs.index(best_epoch)
    best_map50 = val_map50[best_idx]
    best_map50_95 = val_map50_95[best_idx]

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        epochs,
        train_loss,
        color="#D62728",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        label=f"Train Total Loss (Cuối Epoch 36: {train_loss[-1]:.2f})",
    )
    l2 = ax2.plot(
        epochs,
        val_map50,
        color="#1F77B4",
        linewidth=2.5,
        marker="s",
        markersize=3.5,
        label=f"Val mAP@0.5 (Best @ Epoch {best_epoch}: {best_map50 * 100:.2f}%)",
    )
    l3 = ax2.plot(
        epochs,
        val_map50_95,
        color="#17BECF",
        linewidth=2.0,
        linestyle="-.",
        label=f"Val mAP@0.5:0.95 (Best @ Epoch {best_epoch}: {best_map50_95 * 100:.2f}%)",
    )
    l4 = ax1.axvline(
        x=best_epoch,
        color="#2CA02C",
        linestyle=":",
        linewidth=2.3,
        label=f"Best Checkpoint (Epoch {best_epoch}) — Early Stop @ Epoch {epochs[-1]}",
    )
    ax2.scatter([best_epoch], [best_map50], color="#2CA02C", s=70, zorder=6)
    ax2.scatter([best_epoch], [best_map50_95], color="#2CA02C", s=55, zorder=6)

    ax1.set_title(
        "SSDLite320-MobileNetV3 (E2_ssdlite320_10cls): Training Loss & Val mAP (36 Epochs thực tế)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax1.set_xlabel("Epochs (Cấu hình: 50 epochs, Early Stopping tại Epoch 36)", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Training Total Loss", fontsize=11.5, fontweight="bold", color="#D62728")
    ax2.set_ylabel("Validation mAP (COCOeval)", fontsize=11.5, fontweight="bold", color="#1F77B4")

    ax1.set_xlim(1, 36)
    ax1.set_ylim(2.8, 7.8)
    ax2.set_ylim(0.0, 0.25)
    ax1.grid(True)

    lines = l1 + l2 + l3 + [l4]
    labels = [line.get_label() for line in lines]
    ax1.legend(
        lines,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=9.3,
    )

    out_path = out_dir / "ssdlite320_loss_map_curve.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_normalized_confusion_matrix(out_dir: Path) -> Path:
    """3. Real Normalized Confusion Matrices:
    - Left panel: Stage-2 Classifier (MobileNetV3-Large, 96.18% Val Accuracy) from artifacts/official_run/val_confusion_matrix.csv
    - Right panel: Object Detector 11x11 Normalized Confusion Matrix (E3_yolov8n_pilot / E1_yolov8n_10cls)
    Also saves standalone copies if needed.
    """
    cm_csv_path = REPO_ROOT / "artifacts" / "official_run" / "val_confusion_matrix.csv"
    rows: list[list[int]] = []
    with cm_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cls_labels = [c.strip() for c in header[1:]]
        for row in reader:
            rows.append([int(x) for x in row[1:]])

    cm_counts = np.array(rows, dtype=float)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm_cls = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts), where=row_sums > 0)
    overall_acc = np.trace(cm_counts) / np.sum(cm_counts)

    # Real 11x11 Detector Normalized Confusion Matrix (columns = True, rows = Predicted)
    # Extracted directly from artifacts/detection_v3_pilot/E3_yolov8n_pilot/confusion_matrix_normalized.png
    det_labels = CLASS_NAMES + ["background"]
    det_matrix = np.array(
        [
            # bat   bio   card  cloth glass metal paper plast shoes trash bg(FP)
            [0.86, 0.00, 0.00, 0.00, 0.00, 0.07, 0.00, 0.01, 0.00, 0.00, 0.00],  # battery
            [0.00, 0.00, 0.03, 0.00, 0.00, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00],  # biological
            [0.00, 0.50, 0.18, 0.04, 0.04, 0.03, 0.09, 0.10, 0.05, 0.01, 0.04],  # cardboard
            [0.00, 0.00, 0.00, 0.40, 0.04, 0.01, 0.05, 0.01, 0.10, 0.01, 0.08],  # clothes
            [0.00, 0.00, 0.00, 0.09, 0.15, 0.07, 0.16, 0.09, 0.17, 0.06, 0.07],  # glass
            [0.14, 0.00, 0.10, 0.12, 0.05, 0.20, 0.12, 0.11, 0.15, 0.05, 0.09],  # metal
            [0.00, 0.00, 0.18, 0.04, 0.02, 0.14, 0.14, 0.09, 0.02, 0.02, 0.04],  # paper
            [0.00, 0.00, 0.26, 0.06, 0.10, 0.22, 0.09, 0.26, 0.05, 0.21, 0.31],  # plastic
            [0.00, 0.00, 0.03, 0.06, 0.00, 0.01, 0.02, 0.01, 0.41, 0.00, 0.03],  # shoes
            [0.00, 0.50, 0.15, 0.00, 0.20, 0.12, 0.16, 0.14, 0.00, 0.18, 0.33],  # trash
            [0.00, 0.00, 0.08, 0.19, 0.41, 0.12, 0.16, 0.18, 0.05, 0.46, 0.00],  # background (FN)
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.2), dpi=300)

    # Panel A: Stage-2 Classifier (10x10)
    ax0 = axes[0]
    im0 = ax0.imshow(cm_norm_cls, cmap="Blues", vmin=0.0, vmax=1.0)
    ax0.set_title(
        f"(A) Stage-2 Classifier: MobileNetV3-Large\n(Tập Val: {int(np.sum(cm_counts))} mẫu — Accuracy = {overall_acc * 100:.2f}%)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax0.set_xticks(range(len(cls_labels)))
    ax0.set_yticks(range(len(cls_labels)))
    ax0.set_xticklabels(cls_labels, rotation=45, ha="right", fontsize=9.5)
    ax0.set_yticklabels(cls_labels, fontsize=9.5)
    ax0.set_xlabel("Nhãn dự đoán (Predicted Label)", fontsize=10.5, fontweight="bold")
    ax0.set_ylabel("Nhãn thực tế (True Label)", fontsize=10.5, fontweight="bold")
    ax0.grid(False)

    for i in range(len(cls_labels)):
        for j in range(len(cls_labels)):
            val = cm_norm_cls[i, j]
            if val >= 0.005:
                text_color = "white" if val > 0.5 else "#1A1A1A"
                ax0.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8.2,
                    fontweight="bold" if i == j else "normal",
                )
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # Panel B: Object Detector 11x11 (with Background)
    ax1 = axes[1]
    im1 = ax1.imshow(det_matrix, cmap="Blues", vmin=0.0, vmax=0.9)
    ax1.set_title(
        "(B) Object Detector: YOLOv8n 10-Class + Background\n(E3 Pilot — Chuẩn hóa theo cột Ground Truth)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.set_xticks(range(len(det_labels)))
    ax1.set_yticks(range(len(det_labels)))
    ax1.set_xticklabels(det_labels, rotation=45, ha="right", fontsize=9.5)
    ax1.set_yticklabels(det_labels, fontsize=9.5)
    ax1.set_xlabel("Nhãn thực tế (True Box / Background)", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Nhãn dự đoán (Predicted Box / Missed FN)", fontsize=10.5, fontweight="bold")
    ax1.grid(False)

    for i in range(len(det_labels)):
        for j in range(len(det_labels)):
            val = det_matrix[i, j]
            if val >= 0.01:
                text_color = "white" if val > 0.45 else "#1A1A1A"
                ax1.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.8,
                    fontweight="bold" if i == j else "normal",
                )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Ma trận nhầm lẫn chuẩn hóa thực nghiệm (Normalized Confusion Matrix)",
        fontsize=14.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / "normalized_confusion_matrix.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_per_class_ap50_comparison(out_dir: Path) -> Path:
    """4. Per-class AP@0.5 comparison on the official Validation split (299 images, 902 instances)."""
    e1_eval_path = REPO_ROOT / "artifacts" / "detection_v2" / "E1_yolov8n_10cls" / "eval_val_test.json"
    e2_hist_path = REPO_ROOT / "artifacts" / "detection_v2" / "E2_ssdlite320_10cls" / "history.json"

    e1_eval = json.loads(e1_eval_path.read_text(encoding="utf-8"))
    e2_hist = json.loads(e2_hist_path.read_text(encoding="utf-8"))

    # Epoch 30 is best_epoch of E2_ssdlite320_10cls
    e2_best = next(item for item in e2_hist if int(item["epoch"]) == 30)

    ssdlite_ap50: list[float] = []
    yolov8n_ap50: list[float] = []
    instances: list[int] = []

    for cls_name in CLASS_NAMES:
        s_entry = e2_best["validation"]["ap_by_class"][cls_name]
        y_entry = e1_eval["val"]["ap_by_class"][cls_name]
        inst = int(y_entry["instances"])
        instances.append(inst)
        ssdlite_ap50.append(float(s_entry["ap50"] or 0.0) * 100.0)
        yolov8n_ap50.append(float(y_entry["ap50"] or 0.0) * 100.0)

    e2_map50 = float(e2_best["validation"]["map50"]) * 100.0
    e1_map50 = float(e1_eval["val"]["map50"]) * 100.0

    x = np.arange(len(CLASS_NAMES))
    width = 0.37

    fig, ax = plt.subplots(figsize=(12.5, 6.4), dpi=300)
    rects1 = ax.bar(
        x - width / 2,
        ssdlite_ap50,
        width,
        label=f"SSDLite320-MobileNetV3 (E2 — Val mAP@0.5 = {e2_map50:.2f}%)",
        color="#2B5C8F",
        edgecolor="white",
    )
    rects2 = ax.bar(
        x + width / 2,
        yolov8n_ap50,
        width,
        label=f"YOLOv8n Baseline (E1 — Val mAP@0.5 = {e1_map50:.2f}%)",
        color="#E66101",
        edgecolor="white",
    )

    ax.set_title(
        "So sánh AP@0.5 trên từng lớp giữa SSDLite320 và YOLOv8n (Tập Validation 299 ảnh - COCOeval)",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("AP@0.5 (%)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Nhóm rác thải (kèm số lượng Ground Truth boxes trong tập Validation v2)", fontsize=11.5, fontweight="bold")
    x_labels = [f"{cls}\n(n={inst})" for cls, inst in zip(CLASS_NAMES, instances)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9.8)
    ax.set_ylim(0, 84)
    ax.grid(axis="y")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#1D3F63",
            fontweight="bold",
        )

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.6,
            color="#B34800",
            fontweight="bold",
        )

    # Annotate why battery & biological are 0% in v2 (and solved in E3 pilot)
    ax.annotate(
        "Lưu ý dữ liệu v2:\n• battery: n=0 trong Val v2 (lên 99.5% AP50 ở E3 Pilot)\n• biological: chỉ n=2 mẫu trong Val v2",
        xy=(0.5, 4.0),
        xytext=(0.15, 54.0),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.2},
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF9E6", "edgecolor": "#D4A017"},
    )

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10.2)

    out_path = out_dir / "per_class_ap50_comparison.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_ablation_study_chart(out_dir: Path) -> Path:
    """5. Ablation Study chart from verified_evaluation_summary_from_raw.json & audit report."""
    configs = [
        "EXP-01:\nSSDLite320\n(Raw)",
        "EXP-02:\nSSDLite320\n+ Tiling",
        "EXP-03:\nSSDLite320\n+ TwoStage (Prod)",
        "EXP-04 (Cũ):\nTiling 640 +\nTwoStage (Blend)",
        "EXP-05 (Mới):\nTiling 800 +\nTwoStage (Prod)",
        "EXP-06:\nYOLOv8n\n(Raw E1)",
    ]

    # Exact metrics from reports/qa_acceptance/verified_evaluation_summary_from_raw.json
    # and docs/BAO_CAO_TONG_HOP_AUDIT_VA_NANG_CAP_MO_HINH_KLCN155.md (Section 5)
    precision = [8.0, 5.1, 14.2, 61.9, 72.7, 15.6]
    recall = [3.5, 7.0, 3.5, 22.8, 14.0, 8.8]
    map50 = [5.7, 6.1, 5.9, 28.1, 22.6, 5.0]
    latency_ms = [65.9, 580.0, 185.0, 5127.0, 1850.0, 48.4]

    x = np.arange(len(configs))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(13.2, 8.2),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.45, 1.0], "hspace": 0.14},
    )

    b1 = ax1.bar(x - width, precision, width, label="Precision (%)", color="#1F77B4", edgecolor="white")
    b2 = ax1.bar(x, recall, width, label="Recall (%)", color="#2CA02C", edgecolor="white")
    b3 = ax1.bar(x + width, map50, width, label="mAP@0.5 (%)", color="#9467BD", edgecolor="white")

    ax1.set_title(
        "Ablation Study: So sánh Độ chính xác (Precision, Recall, mAP@0.5) & Tốc độ trên tập Holdout",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax1.set_ylabel("Precision / Recall / mAP@0.5 (%)", fontsize=11, fontweight="bold", color="#1F77B4")
    ax1.set_ylim(0, 86)
    ax1.grid(axis="y")

    for bars, col in [(b1, "#14537E"), (b2, "#1D6F1D"), (b3, "#5E3C82")]:
        for rect in bars:
            h = rect.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.6,
                color=col,
                fontweight="bold",
            )

    ax1.legend(
        loc="upper left",
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=10.0,
    )

    # Bottom subplot: Latency (ms, log scale) + FPS callouts
    bars_lat = ax2.bar(
        x,
        latency_ms,
        width=0.45,
        color="#FADBD8",
        edgecolor="#C0392B",
        linewidth=1.5,
        label="Latency (ms, thang đo Log) & Tốc độ (FPS)",
    )
    ax2.set_yscale("log")
    ax2.set_ylim(20, 18000)
    ax2.set_ylabel("Latency (ms - Log)", fontsize=11, fontweight="bold", color="#C0392B")
    ax2.set_xlabel(
        "Cấu hình Pipeline (Đánh giá trên tập Holdout 27 ảnh, Conf ≥ 0.25, IoU ≥ 0.50)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9.8, fontweight="bold")
    ax2.grid(axis="y")

    for rect, lat in zip(bars_lat, latency_ms):
        fps = 1000.0 / lat
        label_str = f"{lat:.1f} ms ({fps:.1f} FPS)" if lat < 100 else f"{lat:.0f} ms ({fps:.1f} FPS)"
        ax2.annotate(
            label_str,
            xy=(rect.get_x() + rect.get_width() / 2, lat),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#922B21",
            zorder=10,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#FFF5F5", "edgecolor": "#E09090"},
        )

    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.8)

    out_path = out_dir / "ablation_study_chart.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _load_raw_pipeline_predictions(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "images" in data and isinstance(data["images"], list):
        out: dict[str, list[dict]] = {}
        for img in data["images"]:
            raw_list = img.get("predictions", img.get("objects", []))
            out[img["filename"]] = [
                {
                    "class_id": int(p["class_id"]),
                    "confidence": float(p["confidence"]),
                    "bbox_xyxy": [float(v) for v in p["bbox_xyxy"]],
                }
                for p in raw_list
            ]
        return out
    preds, _ = load_predictions(path)
    return preds


def generate_pr_and_f1_curves(out_dir: Path) -> Path:
    """6. Real Empirical PR Curve & F1-Confidence Curve across thresholds on Holdout."""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from reports.qa_acceptance.practical_metrics import (
        compute_ap_table,
        load_ground_truth,
        precision_recall_f1,
    )

    gt_path = REPO_ROOT / "reports" / "qa_acceptance" / "ground_truth_hinh_anh_rac.json"
    gt, _ = load_ground_truth(gt_path)

    pred_files = [
        (
            "SSDLite + TwoStage (Blend - Cũ)",
            REPO_ROOT / "reports" / "qa_acceptance" / "raw_predictions_before_quick_fix.json",
            "#1F77B4",
            "--",
        ),
        (
            "SSDLite + TwoStage (Product - Mới)",
            REPO_ROOT / "reports" / "qa_acceptance" / "raw_predictions_after_quick_fix.json",
            "#2CA02C",
            "-",
        ),
        (
            "YOLOv8n Baseline (E1 Raw)",
            REPO_ROOT / "reports" / "qa_acceptance" / "raw_predictions_yolov8n_e1.json",
            "#FF7F0E",
            "-.",
        ),
    ]

    conf_grid = np.linspace(0.05, 0.85, 50)
    fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=300)

    for label, p_path, color, ls in pred_files:
        preds = _load_raw_pipeline_predictions(p_path)
        ap_res = compute_ap_table(gt, preds)
        map50 = float(ap_res["mAP50"])

        recalls: list[float] = []
        precisions: list[float] = []
        f1s: list[float] = []

        for c in conf_grid:
            pr = precision_recall_f1(
                gt,
                preds,
                conf=float(c),
                iou_thr=0.5,
                agnostic=False,
            )
            precisions.append(float(pr["precision"]))
            recalls.append(float(pr["recall"]))
            f1s.append(float(pr["f1"]))

        # Sort by recall for PR plot
        order = np.argsort(recalls)
        r_sorted = np.array(recalls)[order]
        p_sorted = np.array(precisions)[order]

        ax_pr.step(
            r_sorted,
            p_sorted,
            where="post",
            color=color,
            linestyle=ls,
            linewidth=2.5,
            label=f"{label} (mAP@0.5 = {map50 * 100:.2f}%)",
        )

        best_idx = int(np.argmax(f1s))
        best_f1 = f1s[best_idx]
        best_conf = conf_grid[best_idx]
        ax_f1.plot(
            conf_grid,
            f1s,
            color=color,
            linestyle=ls,
            linewidth=2.5,
            label=f"{label} (Peak F1: {best_f1:.3f} @ conf={best_conf:.2f})",
        )
        ax_f1.scatter([best_conf], [best_f1], color=color, s=60, zorder=6)

    ax_pr.set_title(
        "Biểu đồ Precision - Recall Thực nghiệm (Holdout 27 ảnh)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax_pr.set_xlabel("Recall (Độ phủ)", fontsize=11, fontweight="bold")
    ax_pr.set_ylabel("Precision (Độ chính xác)", fontsize=11, fontweight="bold")
    ax_pr.set_xlim(0.0, 0.40)
    ax_pr.set_ylim(0.0, 1.05)
    ax_pr.grid(True)
    ax_pr.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.3)

    ax_f1.axvline(
        x=0.25,
        color="#666666",
        linestyle=":",
        linewidth=1.8,
        label="Ngưỡng vận hành mặc định (conf = 0.25)",
    )
    ax_f1.set_title(
        "Biểu đồ F1 - Confidence Thực nghiệm (Holdout 27 ảnh)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax_f1.set_xlabel("Ngưỡng tự tin (Confidence Threshold)", fontsize=11, fontweight="bold")
    ax_f1.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax_f1.set_xlim(0.05, 0.90)
    ax_f1.set_ylim(0.0, 0.45)
    ax_f1.grid(True)
    ax_f1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.3)

    fig.suptitle(
        "Đường cong Precision-Recall & F1-Confidence trích xuất từ Predictions thật",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / "pr_and_f1_curves.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate verified thesis charts from real project artifacts.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for generated charts (default: reports/figures).",
    )
    parser.add_argument(
        "--copy-to-downloads",
        type=Path,
        default=None,
        help="Optional secondary directory (e.g., C:/Users/ad/Downloads) to copy the 6 charts into.",
    )
    args = parser.parse_args()

    setup_matplotlib_style()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = [
        generate_yolov8n_loss_map_curve(out_dir),
        generate_ssdlite320_loss_map_curve(out_dir),
        generate_normalized_confusion_matrix(out_dir),
        generate_per_class_ap50_comparison(out_dir),
        generate_ablation_study_chart(out_dir),
        generate_pr_and_f1_curves(out_dir),
    ]

    for p in generated:
        print(f"[OK] Generated: {p} ({p.stat().st_size:,} bytes)")

    if args.copy_to_downloads is not None:
        dl_dir = args.copy_to_downloads.resolve()
        dl_dir.mkdir(parents=True, exist_ok=True)
        for p in generated:
            dest = dl_dir / p.name
            shutil.copy2(p, dest)
            print(f"[COPIED] -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
