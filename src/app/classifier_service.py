"""High-precision 10-class waste classifier using MobileNetV3-Large (ONNX / PyTorch)."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.detection.schema import CLASS_NAMES

VI_LABELS = {
    "battery": "Pin / Ắc quy (Rác nguy hại)",
    "biological": "Rác hữu cơ / Thực phẩm",
    "cardboard": "Bìa carton (Tái chế)",
    "clothes": "Quần áo / Vải cũ",
    "glass": "Thủy tinh (Tái chế)",
    "metal": "Kim loại / Lon nhôm (Tái chế)",
    "paper": "Giấy / Báo cũ (Tái chế)",
    "plastic": "Nhựa / Chai túi nilon (Tái chế)",
    "shoes": "Giày dép cũ",
    "trash": "Rác vô cơ khác (Túi rác/hỗn hợp)",
}

BIN_RECOMMENDATIONS = {
    "biological": ("Thùng Xanh lá (Rác hữu cơ)", "Thức ăn thừa, vỏ rau củ quả, bã trà, lá cây."),
    "plastic": ("Thùng Vàng / Trắng (Rác tái chế)", "Chai nhựa, túi nilon sạch, hộp xốp sạch, ly nhựa."),
    "paper": ("Thùng Vàng / Trắng (Rác tái chế)", "Giấy vụn, thùng carton, sách báo cũ."),
    "cardboard": ("Thùng Vàng / Trắng (Rác tái chế)", "Bìa carton, hộp giấy carton sóng."),
    "metal": ("Thùng Vàng / Trắng (Rác tái chế)", "Lon nước ngọt, vỏ đồ hộp kim loại, dây đồng."),
    "glass": ("Thùng Vàng / Trắng (Rác tái chế)", "Chai lọ thủy tinh, ly chén thủy tinh vỡ."),
    "battery": ("Thùng Đỏ / Cam (Rác nguy hại)", "Pin tiểu, pin điện thoại, ắc quy, bóng đèn huỳnh quang."),
    "clothes": ("Thùng Xám / Tái sử dụng", "Quần áo cũ còn dùng được (quyên góp) hoặc vải vụn."),
    "shoes": ("Thùng Xám / Tái sử dụng", "Giày dép cũ hư hỏng."),
    "trash": ("Thùng Xám (Rác vô cơ khác)", "Rác thải sinh hoạt không thể tái chế, túi rác hỗn hợp."),
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def summarize_bin_guidance(class_ids: Iterable[str]) -> list[tuple[str, str]]:
    """Return deduplicated (message, tone) waste-sorting instructions for detected classes."""
    classes = set(class_ids)
    guidance: list[tuple[str, str]] = []
    if "biological" in classes:
        guidance.append(("**Thùng Xanh lá (Rác hữu cơ):** Thức ăn thừa, rau củ quả, lá cây.", "success"))
    if classes.intersection({"plastic", "paper", "cardboard", "metal", "glass"}):
        guidance.append(
            ("**Thùng Vàng / Trắng (Rác tái chế):** Nhựa, giấy, bìa carton, kim loại, thủy tinh.", "success")
        )
    if "battery" in classes:
        guidance.append(("**Thùng Đỏ / Cam (Rác nguy hại):** Pin, ắc quy, bóng đèn, rác độc hại.", "error"))
    if classes.intersection({"clothes", "shoes", "trash"}):
        guidance.append(
            (
                "**Thùng Xám / Thu gom riêng:** Quần áo cũ, giày dép (tái sử dụng/quyên góp), rác vô cơ khác.",
                "neutral",
            )
        )
    return guidance


@dataclass(frozen=True)
class ClassificationOutput:
    class_id: str
    class_name_vi: str
    confidence: float
    top_probabilities: list[tuple[str, str, float]]
    bin_name: str
    bin_guide: str
    processing_ms: float


class MobileNetClassifier:
    """Single-object 10-class MobileNetV3-Large classifier with ONNX Runtime acceleration."""

    def __init__(
        self,
        checkpoint_path: Path | str = "artifacts/official_run/best_model.pt",
        device: str = "auto",
        prefer_onnx: bool = True,
    ) -> None:
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {p}")

        self.class_names: list[str] = list(CLASS_NAMES)
        self._onnx_session = None
        self._onnx_input_name = ""
        self.model = None
        self.device = None
        self.transform = None

        onnx_candidate = p.with_suffix(".onnx") if p.suffix.lower() == ".onnx" else p.parent / "mobilenetv3_large_waste.onnx"
        if prefer_onnx and onnx_candidate.is_file():
            try:
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                if device in {"auto", "cuda"} and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._onnx_session = ort.InferenceSession(str(onnx_candidate), providers=providers)
                self._onnx_input_name = self._onnx_session.get_inputs()[0].name
                return
            except Exception:
                self._onnx_session = None

        import torch
        import torchvision.models as models
        from torchvision.transforms import v2

        resolved_device = torch.device(
            "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
        )
        ckpt = torch.load(p, map_location=resolved_device, weights_only=False)
        self.class_names = list(ckpt.get("class_names", CLASS_NAMES))
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(self.class_names))
        model.load_state_dict(ckpt["state_dict"])
        model.to(resolved_device)
        model.eval()
        self.model = model
        self.device = resolved_device
        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _predict_probabilities(self, rgb: Image.Image) -> np.ndarray:
        if self._onnx_session is not None:
            resized = rgb.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            norm = ((arr - _IMAGENET_MEAN) / _IMAGENET_STD)[np.newaxis, ...]
            logits = self._onnx_session.run(None, {self._onnx_input_name: norm})[0][0]
            shifted = logits - np.max(logits)
            exp_scores = np.exp(shifted)
            return exp_scores / np.sum(exp_scores)

        import torch

        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.model(tensor)
            return torch.softmax(out, dim=1)[0].cpu().numpy()

    def classify_pil(self, image: Image.Image) -> ClassificationOutput:
        started = time.perf_counter()
        probs = self._predict_probabilities(image.convert("RGB"))
        top_indices = np.argsort(probs)[::-1][: min(5, len(self.class_names))]
        top_probs = [
            (
                self.class_names[int(idx)],
                VI_LABELS.get(self.class_names[int(idx)], self.class_names[int(idx)]),
                float(probs[int(idx)]),
            )
            for idx in top_indices
        ]
        best_id, best_vi, best_conf = top_probs[0]
        bin_name, bin_guide = BIN_RECOMMENDATIONS.get(
            best_id, ("Thùng thông thường", "Phân loại theo quy định tại nguồn.")
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ClassificationOutput(
            class_id=best_id,
            class_name_vi=best_vi,
            confidence=best_conf,
            top_probabilities=top_probs,
            bin_name=bin_name,
            bin_guide=bin_guide,
            processing_ms=elapsed_ms,
        )
