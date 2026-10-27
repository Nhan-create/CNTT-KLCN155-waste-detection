"""High-precision 10-class waste classifier using MobileNetV3-Large (96.18% Accuracy)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.models as models
from PIL import Image
from torchvision.transforms import v2

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
    "biological": ("🟢 Thùng Xanh lá (Rác hữu cơ)", "Thức ăn thừa, vỏ rau củ quả, bã trà, lá cây."),
    "plastic": ("🟡 Thùng Vàng / Trắng (Rác tái chế)", "Chai nhựa, túi nilon sạch, hộp xốp sạch, ly nhựa."),
    "paper": ("🟡 Thùng Vàng / Trắng (Rác tái chế)", "Giấy vụn, thùng carton, sách báo cũ."),
    "cardboard": ("🟡 Thùng Vàng / Trắng (Rác tái chế)", "Bìa carton, hộp giấy carton sóng."),
    "metal": ("🟡 Thùng Vàng / Trắng (Rác tái chế)", "Lon nước ngọt, vỏ đồ hộp kim loại, dây đồng."),
    "glass": ("🟡 Thùng Vàng / Trắng (Rác tái chế)", "Chai lọ thủy tinh, ly chén thủy tinh vỡ."),
    "battery": ("🔴 Thùng Đỏ / Cam (Rác nguy hại)", "Pin tiểu, pin điện thoại, ắc quy, bóng đèn huỳnh quang."),
    "clothes": ("⚪ Thùng Xám / Tái sử dụng", "Quần áo cũ còn dùng được (quyên góp) hoặc vải vụn."),
    "shoes": ("⚪ Thùng Xám / Tái sử dụng", "Giày dép cũ hư hỏng."),
    "trash": ("⚪ Thùng Xám (Rác vô cơ khác)", "Rác thải sinh hoạt không thể tái chế, túi rác hỗn hợp."),
}


@dataclass(frozen=True)
class ClassificationOutput:
    class_id: str
    class_name_vi: str
    confidence: float
    top_probabilities: list[tuple[str, str, float]]  # (class_id, class_name_vi, prob)
    bin_name: str
    bin_guide: str
    processing_ms: float


class MobileNetClassifier:
    def __init__(
        self,
        checkpoint_path: Path | str = "artifacts/official_run/best_model.pt",
        device: str = "auto",
    ) -> None:
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Classifier checkpoint not found: {p}")
        resolved_device = torch.device(
            "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
        )
        ckpt = torch.load(p, map_location=resolved_device, weights_only=False)
        self.class_names = ckpt.get(
            "class_names",
            [
                "battery", "biological", "cardboard", "clothes", "glass",
                "metal", "paper", "plastic", "shoes", "trash"
            ],
        )
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

    def classify_pil(self, image: Image.Image) -> ClassificationOutput:
        started = time.perf_counter()
        rgb = image.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0].cpu()

        topk = torch.topk(probs, min(5, len(self.class_names)))
        top_probs = [
            (
                self.class_names[idx],
                VI_LABELS.get(self.class_names[idx], self.class_names[idx]),
                probs[idx].item(),
            )
            for idx in topk.indices
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
