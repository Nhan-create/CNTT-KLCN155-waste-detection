from pathlib import Path

import pytest

from src.web.detection_settings import (
    DEFAULT_SSDLITE_PATH,
    DEFAULT_YOLO_PATH,
    detector_problem,
    parse_detection_settings,
    paths_for_backend,
)


def test_default_is_phase2_ssdlite() -> None:
    settings = parse_detection_settings([], {})

    assert settings.backend == "ssdlite"
    assert settings.model_path == DEFAULT_SSDLITE_PATH
    assert settings.secondary_model_path is None


def test_yolov8n_uses_its_own_checkpoint_path() -> None:
    settings = parse_detection_settings(["--backend", "yolov8n"], {})

    assert settings.model_path == DEFAULT_YOLO_PATH


def test_fusion_uses_both_trained_checkpoints(tmp_path: Path) -> None:
    primary = tmp_path / "ssdlite.pt"
    secondary = tmp_path / "yolov8n.pt"
    primary.touch()
    settings = parse_detection_settings(
        ["--backend", "wbf", "--detector", str(primary), "--secondary-detector", str(secondary)],
        {},
    )

    assert "hai mô hình" in detector_problem(settings)
    secondary.touch()
    assert detector_problem(settings) is None


def test_backend_switch_keeps_custom_weights() -> None:
    settings = parse_detection_settings(
        ["--backend", "wbf", "--detector", "primary.pt", "--secondary-detector", "second.pt"],
        {},
    )

    assert paths_for_backend(settings, "ssdlite") == (Path("primary.pt"), None)
    assert paths_for_backend(settings, "yolov8n") == (Path("second.pt"), None)
    assert paths_for_backend(settings, "wbf") == (Path("primary.pt"), Path("second.pt"))


def test_environment_configuration_and_cli_priority() -> None:
    settings = parse_detection_settings(
        ["--detector", "cli.pt"],
        {
            "WASTE_DETECTOR_BACKEND": "wbf",
            "WASTE_DETECTOR": "env.pt",
            "WASTE_SECONDARY_DETECTOR": "env-second.pt",
        },
    )

    assert settings.backend == "wbf"
    assert settings.model_path == Path("cli.pt")
    assert settings.secondary_model_path == Path("env-second.pt")


@pytest.mark.parametrize("argv", [["--backend", "yolo26s"], ["--confidence-threshold", "nan"]])
def test_invalid_phase2_options_are_rejected(argv: list[str]) -> None:
    with pytest.raises(ValueError):
        parse_detection_settings(argv, {})


def test_cli_detector_and_realtime_settings_are_parsed() -> None:
    settings = parse_detection_settings(
        [
            "--detector",
            "custom.pt",
            "--confidence-threshold",
            "0.42",
            "--live-inference-fps",
            "7",
        ],
        {},
    )

    assert settings.model_path == Path("custom.pt")
    assert settings.confidence_threshold == 0.42
    assert settings.live_inference_fps == 7
