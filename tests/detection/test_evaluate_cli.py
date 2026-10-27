# Mô tả: Kiểm thử giao diện dòng lệnh CLI đánh giá bộ phân loại (python -m src.detection.classifier_evaluate).
import subprocess
import sys


def test_evaluate_cli_help_lists_checkpoint_and_test_root() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "src.detection.classifier_evaluate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--checkpoint" in completed.stdout
    assert "--test-root" in completed.stdout
    assert "--output-dir" in completed.stdout
