"""Regression checks for training on real scenes and empty background frames."""

from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn
from torchvision.models.detection._utils import BoxCoder

from src.data.detection_manifest import prepare_detection_manifest
from src.detection.evaluation import ImagePrediction, summarize_predictions
from src.detection.schema import DETECTION_CLASS_NAMES
from src.detection.ssdlite import ARCHITECTURE, build_ssdlite, tensor_detections
from src.detection.training import (
    DetectionTrainingDataset, TrainingConfig, evaluate_detector, run_training, ssd_loss_with_background,
)
from src.detection.types import BoundingBox, Detection


def test_empty_image_has_nonzero_background_loss_and_gradients():
    model = SimpleNamespace(
        box_coder=BoxCoder(weights=(10.0, 10.0, 5.0, 5.0)),
        neg_to_pos_ratio=3,
        minimum_background_samples=2,
    )
    logits = torch.tensor([[[0.0, 2.0], [0.0, 1.0], [3.0, 0.0]]], requires_grad=True)
    regression = torch.zeros((1, 3, 4), requires_grad=True)
    anchors = [torch.tensor([[0., 0., 10., 10.], [10., 10., 20., 20.], [20., 20., 30., 30.]])]
    targets = [{"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}]

    losses = ssd_loss_with_background(
        model, targets, {"cls_logits": logits, "bbox_regression": regression},
        anchors, [torch.full((3,), -1, dtype=torch.int64)],
    )
    sum(losses.values()).backward()

    assert losses["classification"].item() > 0
    assert losses["bbox_regression"].item() == 0
    assert logits.grad[0, :2].abs().sum().item() > 0
    assert logits.grad[0, 2].abs().sum().item() == 0


def test_training_retains_original_size_and_background_images(tmp_path: Path):
    image_path = tmp_path / "background.png"
    labels_path = tmp_path / "background.txt"
    Image.new("RGB", (640, 240), "green").save(image_path)
    labels_path.write_text("", encoding="utf-8")
    dataset = DetectionTrainingDataset([
        {"image": str(image_path), "labels": str(labels_path), "source": "field"}
    ])

    image, target, record = dataset[0]

    assert image.shape == (3, 240, 640)
    assert image.dtype == torch.float32
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert record["source"] == "field"


def test_normalized_boxes_follow_original_image_dimensions(tmp_path: Path):
    image_path = tmp_path / "scene.png"
    labels_path = tmp_path / "scene.txt"
    Image.new("RGB", (600, 200), "blue").save(image_path)
    labels_path.write_text("7 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    dataset = DetectionTrainingDataset([
        {"image": str(image_path), "labels": str(labels_path), "source": "field"}
    ])

    _, target, _ = dataset[0]

    torch.testing.assert_close(target["boxes"], torch.tensor([[150., 50., 450., 150.]]))
    assert target["labels"].tolist() == [8]


def test_evaluation_requires_correct_class_and_one_match_per_object():
    class Detector(nn.Module):
        def forward(self, images):
            return [{
                "boxes": torch.tensor([[10., 10., 30., 30.], [10., 10., 30., 30.], [50., 50., 70., 70.]]),
                "labels": torch.tensor([8, 8, 1]),
                "scores": torch.tensor([.9, .8, .7]),
            }]

    batch = ([torch.zeros(3, 100, 100)], [{
        "boxes": torch.tensor([[10., 10., 30., 30.], [50., 50., 70., 70.]]),
        "labels": torch.tensor([8, 5]),
    }], [{"source": "field"}])

    report = evaluate_detector(Detector(), [batch], torch.device("cpu"), confidence_threshold=.29)

    assert report["true_positives"] == 1
    assert report["predictions"] == 3
    assert report["objects"] == 2
    assert report["precision"] == pytest.approx(1 / 3)
    assert report["recall"] == .5
    assert report["per_class"]["glass"]["recall"] == 0
    assert report["per_source"]["field"]["true_positives"] == 1


def test_evaluation_shares_runtime_decoding_and_stable_equal_iou_matching():
    output = {
        "boxes": torch.tensor([[0., 0., 0., 10.], [0., 0., 20., 10.], [0., 0., 10., 10.]]),
        "labels": torch.tensor([8, 8, 8]),
        "scores": torch.tensor([.99, .9, .8]),
    }

    class Detector(nn.Module):
        def forward(self, images):
            return [output]

    target = {
        "boxes": torch.tensor([[0., 0., 10., 10.], [10., 0., 20., 10.]]),
        "labels": torch.tensor([8, 8]),
    }
    batch = ([torch.zeros(3, 10, 20)], [target], [{"source": "field"}])
    sample = ImagePrediction(
        Path("scene.jpg"), "field", (
            Detection(7, "plastic", 1., BoundingBox(0, 0, 10, 10)),
            Detection(7, "plastic", 1., BoundingBox(10, 0, 20, 10)),
        ), tensor_detections(output, 20, 10, confidence_threshold=.29),
    )

    trained = evaluate_detector(Detector(), [batch], torch.device("cpu"), confidence_threshold=.29)
    runtime = summarize_predictions([sample], default_threshold=.29)

    # The broad box first claims target 1 on the IoU tie. The remaining narrow
    # box cannot claim target 2. The zero-width row contributes no prediction.
    assert trained["predictions"] == runtime["predictions"] == 2
    assert trained["true_positives"] == runtime["true_positives"] == 1
    assert trained["precision"] == runtime["precision"] == .5
    assert trained["recall"] == runtime["recall"] == .5


def test_training_cli_saves_selected_validation_model_and_resumes(tmp_path: Path):
    torch.set_num_threads(2)
    source = tmp_path / "source"
    for split_index, split in enumerate(("train", "val", "test")):
        for index in range(2):
            image_path = source / "images" / split / f"field_{split}_{index}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (80 + index, 60), (20 + split_index * 50, 30 + index * 40, 60)).save(image_path)
            label = source / "labels" / split / image_path.with_suffix(".txt").name
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text("7 .5 .5 .5 .5\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    manifest = prepare_detection_manifest(source, prepared)
    initial = tmp_path / "initial.pt"
    torch.manual_seed(17)
    torch.save({
        "epoch": 3, "model_state_dict": build_ssdlite(pretrained=False).state_dict(),
        "metadata": {
            "architecture": ARCHITECTURE, "class_names": list(DETECTION_CLASS_NAMES),
            "backend": "ssdlite", "background_index": 0, "trained": True,
        },
    }, initial)
    output = tmp_path / "run"
    command = [
        sys.executable, "-m", "scripts.train_ssdlite", "--dataset", str(prepared / "dataset.json"),
        "--output", str(output), "--initial-weights", str(initial), "--epochs", "1",
        "--batch-size", "2", "--device", "cpu", "--threads", "2",
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((output / "test_metrics.json").read_text())
    best = torch.load(output / "weights" / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(output / "weights" / "last.pt", map_location="cpu", weights_only=True)

    assert report["dataset_fingerprint"] == manifest["fingerprint"]
    assert report["test"]["images"] == 2
    assert report["validation"]["map50_95"] == max(row["validation"]["map50_95"] for row in last["history"])
    assert best["metadata"]["validation_domain"] == "real"
    assert best["metadata"]["dataset_manifest"] == str((prepared / "dataset.json").resolve())
    assert last["training_epoch"] == 1
    assert {"optimizer", "scheduler", "rng"} <= last.keys()
    best_bytes = (output / "weights" / "best.pt").read_bytes()

    resume_config = TrainingConfig(
        prepared / "dataset.json", output, initial, epochs=1, batch_size=2, resume=True,
    )
    resumed = run_training(resume_config)
    assert resumed["validation"] == report["validation"]
    assert (output / "weights" / "best.pt").read_bytes() == best_bytes
    # Simulate interruption after writing best.pt but before committing last.pt.
    (output / "weights" / "best.pt").write_bytes(b"interrupted best checkpoint")
    recovered = run_training(resume_config)
    restored = torch.load(output / "weights" / "best.pt", map_location="cpu", weights_only=True)
    for name, value in best["model_state_dict"].items():
        torch.testing.assert_close(restored["model_state_dict"][name], value)
    assert recovered["validation"] == report["validation"]
    with pytest.raises(FileExistsError):
        run_training(TrainingConfig(prepared / "dataset.json", output, initial, epochs=1, batch_size=2))
