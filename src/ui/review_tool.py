"""
src/ui/review_tool.py
---------------------
Streamlit Review & Verification Tool for Multi-Object Waste Detection Bounding Boxes.

Features:
- Visual inspection of bounding boxes on Real (OpenImages) and Synthetic (Mendeley) images.
- Full CRUD for bounding boxes: Add new box, Edit coordinates/class, Delete box.
- Validation checks for normalized YOLO coordinates [0, 1] and class taxonomy [0, 9].
- Status progression: UNREVIEWED -> APPROVED / REJECTED / NEEDS_RELABEL.
- Persists changes directly to YOLO .txt label files and manifest_detection_v1.csv.
- Maintains an append-only audit log: data/detection/review_audit_log.csv.
"""

import os
import sys
import json
import datetime
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Setup Project Root & Data Directory (Configurable via env var for test sandboxing)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_data_dir() -> Path:
    env_dir = os.environ.get("REVIEW_TOOL_DATA_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.exists():
            return p
    return PROJECT_ROOT / "data" / "detection"

DATA_DIR = get_data_dir()
MANIFEST_PATH = DATA_DIR / "manifest_detection_v1.csv"
AUDIT_LOG_PATH = DATA_DIR / "review_audit_log.csv"

TAXONOMY_10 = [
    "battery",      # 0
    "biological",   # 1
    "cardboard",    # 2
    "clothes",      # 3
    "glass",        # 4
    "metal",        # 5
    "paper",        # 6
    "plastic",      # 7
    "shoes",        # 8
    "trash"         # 9
]

CLASS_COLORS = [
    "#E6194B",  # battery - red
    "#3CBA54",  # biological - green
    "#9A6324",  # cardboard - brown
    "#F58231",  # clothes - orange
    "#4363D8",  # glass - blue
    "#911EB4",  # metal - purple
    "#46F0F0",  # paper - cyan
    "#F032E6",  # plastic - magenta
    "#BCF60C",  # shoes - lime
    "#808080",  # trash - gray
]

st.set_page_config(
    page_title="Waste Detection BBox Review Tool",
    page_icon="📦",
    layout="wide"
)

def validate_box(b: dict) -> tuple[bool, str]:
    """Strict geometric and taxonomy validation for a bounding box."""
    cls_id = b.get("class_id")
    if cls_id is None or not (0 <= int(cls_id) <= 9):
        return False, f"Class ID {cls_id} out of bounds [0, 9]"
    xc, yc = float(b.get("xc", 0.0)), float(b.get("yc", 0.0))
    w, h = float(b.get("w", 0.0)), float(b.get("h", 0.0))
    if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0):
        return False, f"Center coordinates ({xc:.4f}, {yc:.4f}) must be within [0.0, 1.0]"
    if w <= 0.001 or h <= 0.001:
        return False, f"Box dimensions w={w:.4f}, h={h:.4f} too small (<= 0.001)"
    if w > 1.0 or h > 1.0:
        return False, f"Box dimensions w={w:.4f}, h={h:.4f} exceed 1.0"
    xmin = xc - w / 2.0
    xmax = xc + w / 2.0
    ymin = yc - h / 2.0
    ymax = yc + h / 2.0
    if xmin < -0.001 or xmax > 1.001 or ymin < -0.001 or ymax > 1.001:
        return False, f"Box boundaries [{xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f}] exceed image boundaries [0, 1]"
    if w * h < 0.00005:
        return False, f"Box area ({w*h:.6f}) is smaller than minimum threshold (0.00005)"
    return True, ""

def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        st.error(f"Manifest file not found: {MANIFEST_PATH}")
        st.stop()
    return pd.read_csv(MANIFEST_PATH)

def save_manifest(df: pd.DataFrame):
    df.to_csv(MANIFEST_PATH, index=False)

def log_audit(image_id: str, old_status: str, new_status: str, num_boxes: int, notes: str, action: str = "STATUS_UPDATE"):
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not AUDIT_LOG_PATH.exists()
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        if header:
            f.write("timestamp,image_id,action,old_status,new_status,num_boxes,notes\n")
        ts = datetime.datetime.now().isoformat()
        clean_notes = notes.replace(",", ";").replace("\n", " ")
        f.write(f"{ts},{image_id},{action},{old_status},{new_status},{num_boxes},{clean_notes}\n")

def read_yolo_boxes(label_path: Path):
    if not label_path.exists():
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append({
                        "class_id": cls_id,
                        "class_name": TAXONOMY_10[cls_id] if 0 <= cls_id < 10 else f"class_{cls_id}",
                        "xc": xc,
                        "yc": yc,
                        "w": w,
                        "h": h
                    })
                except ValueError:
                    pass
    return boxes

def write_yolo_boxes(label_path: Path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(f"{b['class_id']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

def draw_boxes_on_image(img: Image.Image, boxes):
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    for idx, b in enumerate(boxes):
        cls_id = b["class_id"]
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        xc, yc, w, h = b["xc"], b["yc"], b["w"], b["h"]

        xmin = (xc - w / 2) * img_w
        ymin = (yc - h / 2) * img_h
        xmax = (xc + w / 2) * img_w
        ymax = (yc + h / 2) * img_h

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Label text
        label_text = f"#{idx+1}: {b['class_name']} ({cls_id})"
        # Draw background banner for label
        banner_h = 16
        draw.rectangle([xmin, max(0, ymin - banner_h), xmin + len(label_text) * 8 + 6, max(0, ymin)], fill=color)
        draw.text((xmin + 3, max(0, ymin - banner_h) + 1), label_text, fill="white")

    return img

def main():
    st.title("📦 Multi-Object Waste BBox Review Tool (CNTT-KLCN155)")
    st.caption("Standalone in-repo tool for auditing bounding boxes, fixing coordinates/classes, and validating datasets.")

    df = load_manifest()

    # --- SIDEBAR: Filters & Progress ---
    st.sidebar.header("🔍 Dataset Filters")

    # Source filter
    source_choice = st.sidebar.selectbox("Filter Source", ["All", "Real (OpenImages)", "Synthetic (Mendeley)"])
    filtered_df = df.copy()
    if source_choice == "Real (OpenImages)":
        filtered_df = filtered_df[~filtered_df["is_synthetic"]]
    elif source_choice == "Synthetic (Mendeley)":
        filtered_df = filtered_df[filtered_df["is_synthetic"]]

    # Status filter
    status_choice = st.sidebar.selectbox(
        "Filter Review Status",
        ["All", "UNREVIEWED", "APPROVED", "REJECTED", "NEEDS_RELABEL"]
    )
    if status_choice != "All":
        filtered_df = filtered_df[filtered_df["review_status"] == status_choice]

    st.sidebar.divider()
    st.sidebar.subheader("📊 Review Progress")
    total_imgs = len(df)
    rev_approved = sum(df["review_status"] == "APPROVED")
    rev_rejected = sum(df["review_status"] == "REJECTED")
    rev_relabel = sum(df["review_status"] == "NEEDS_RELABEL")
    rev_pending = sum(df["review_status"] == "UNREVIEWED")

    st.sidebar.metric("Total Images", total_imgs)
    st.sidebar.metric("Approved", f"{rev_approved} ({rev_approved/total_imgs*100:.1f}%)")
    st.sidebar.metric("Rejected", rev_rejected)
    st.sidebar.metric("Needs Relabel", rev_relabel)
    st.sidebar.metric("Pending Unreviewed", rev_pending)

    if filtered_df.empty:
        st.warning("No images match the selected filter criteria.")
        return

    # Image selector
    image_list = filtered_df["image_id"].tolist()
    selected_img_id = st.sidebar.selectbox("Select Image ID", image_list, index=0)

    # --- MAIN VIEW ---
    row = df[df["image_id"] == selected_img_id].iloc[0]
    img_rel_path = row["relative_image_path"]
    lbl_rel_path = row["relative_label_path"]
    img_full_path = DATA_DIR / img_rel_path
    lbl_full_path = DATA_DIR / lbl_rel_path

    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    col_meta1.metric("Image ID", row["image_id"])
    col_meta2.metric("Source", f"{row['source']} {'(Synthetic)' if row['is_synthetic'] else '(Real)'}")
    col_meta3.metric("Original Split", row["original_split"])
    col_meta4.metric("Current Status", row["review_status"])

    # Load boxes and image
    if "current_boxes" not in st.session_state or st.session_state.get("active_img_id") != selected_img_id:
        st.session_state.active_img_id = selected_img_id
        st.session_state.current_boxes = read_yolo_boxes(lbl_full_path)

    boxes = st.session_state.current_boxes

    # Image column & Controls column
    col_img, col_ctrl = st.columns([3, 2])

    with col_img:
        if img_full_path.exists():
            img = Image.open(img_full_path).convert("RGB")
            annotated_img = draw_boxes_on_image(img.copy(), boxes)
            st.image(annotated_img, caption=f"{row['filename']} ({img.width}x{img.height}) - {len(boxes)} boxes", use_container_width=True)
        else:
            st.error(f"Image not found at: {img_full_path}")

    with col_ctrl:
        st.subheader("📝 Bounding Boxes Management")

        # Edit existing boxes
        box_to_delete = None
        for i, b in enumerate(boxes):
            with st.expander(f"Box #{i+1}: {b['class_name']}", expanded=(i == 0)):
                c1, c2 = st.columns([1, 1])
                new_cls = c1.selectbox(
                    f"Class #{i+1}",
                    range(len(TAXONOMY_10)),
                    index=b["class_id"],
                    format_func=lambda cid: f"{cid}: {TAXONOMY_10[cid]}",
                    key=f"cls_{selected_img_id}_{i}"
                )
                b["class_id"] = new_cls
                b["class_name"] = TAXONOMY_10[new_cls]

                b["xc"] = c2.number_input(f"Center X #{i+1}", 0.0, 1.0, float(b["xc"]), step=0.01, format="%.4f", key=f"xc_{selected_img_id}_{i}")
                b["yc"] = c1.number_input(f"Center Y #{i+1}", 0.0, 1.0, float(b["yc"]), step=0.01, format="%.4f", key=f"yc_{selected_img_id}_{i}")
                b["w"] = c2.number_input(f"Width #{i+1}", 0.001, 1.0, float(b["w"]), step=0.01, format="%.4f", key=f"w_{selected_img_id}_{i}")
                b["h"] = c1.number_input(f"Height #{i+1}", 0.001, 1.0, float(b["h"]), step=0.01, format="%.4f", key=f"h_{selected_img_id}_{i}")

                if c2.button(f"🗑️ Delete Box #{i+1}", key=f"del_{selected_img_id}_{i}"):
                    box_to_delete = i

        if box_to_delete is not None:
            boxes.pop(box_to_delete)
            st.session_state.current_boxes = boxes
            st.rerun()

        # Add New Box
        with st.expander("➕ Add New Bounding Box"):
            add_c1, add_c2 = st.columns(2)
            add_cls = add_c1.selectbox(
                "New Box Class",
                range(len(TAXONOMY_10)),
                format_func=lambda cid: f"{cid}: {TAXONOMY_10[cid]}",
                key="add_cls"
            )
            add_xc = add_c2.number_input("Center X", 0.0, 1.0, 0.5, step=0.01, format="%.4f", key="add_xc")
            add_yc = add_c1.number_input("Center Y", 0.0, 1.0, 0.5, step=0.01, format="%.4f", key="add_yc")
            add_w = add_c2.number_input("Width", 0.001, 1.0, 0.2, step=0.01, format="%.4f", key="add_w")
            add_h = add_c1.number_input("Height", 0.001, 1.0, 0.2, step=0.01, format="%.4f", key="add_h")

            if st.button("Add Box to Image"):
                candidate_box = {
                    "class_id": add_cls,
                    "class_name": TAXONOMY_10[add_cls],
                    "xc": add_xc,
                    "yc": add_yc,
                    "w": add_w,
                    "h": add_h
                }
                is_valid, err_msg = validate_box(candidate_box)
                if not is_valid:
                    st.error(f"❌ Cannot add box: {err_msg}")
                else:
                    boxes.append(candidate_box)
                    st.session_state.current_boxes = boxes
                    st.rerun()

        st.divider()
        st.subheader("🎯 Review Decision")
        action_decision = st.radio(
            "Set Verification Status",
            ["APPROVED", "REJECTED", "NEEDS_RELABEL"],
            index=0 if row["review_status"] == "APPROVED" else (1 if row["review_status"] == "REJECTED" else 0),
            horizontal=True
        )
        notes = st.text_input("Reviewer Notes", value="Verified bounding boxes and class labels.")

        if st.button("💾 Save Changes & Update Manifest", type="primary", use_container_width=True):
            # Validate all boxes before saving
            validation_errors = []
            for idx, b in enumerate(boxes):
                valid, err = validate_box(b)
                if not valid:
                    validation_errors.append(f"Box #{idx+1}: {err}")

            if validation_errors:
                st.error("❌ Validation Failed! Cannot save changes:\n" + "\n".join(validation_errors))
            else:
                # Backup existing file before overwrite
                if lbl_full_path.exists():
                    backup_path = lbl_full_path.with_suffix(".txt.bak")
                    try:
                        shutil.copy2(lbl_full_path, backup_path)
                    except Exception:
                        pass

                # Save label file
                write_yolo_boxes(lbl_full_path, boxes)

                # Update manifest dataframe
                old_status = row["review_status"]
                cls_counts = {}
                for b in boxes:
                    cls_counts[b["class_id"]] = cls_counts.get(b["class_id"], 0) + 1

                idx_in_df = df[df["image_id"] == selected_img_id].index[0]
                df.at[idx_in_df, "review_status"] = action_decision
                df.at[idx_in_df, "num_boxes"] = len(boxes)
                df.at[idx_in_df, "class_distribution"] = json.dumps(cls_counts)
                save_manifest(df)

                # Audit log
                log_audit(selected_img_id, old_status, action_decision, len(boxes), notes, action="SAVE_MANUAL")

                st.success(f"Successfully saved {selected_img_id} as {action_decision} ({len(boxes)} boxes)!")
                st.rerun()

if __name__ == "__main__":
    main()
