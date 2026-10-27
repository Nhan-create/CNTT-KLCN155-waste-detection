import importlib
import inspect
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "module",
    [
        "src.data.schema",
        "src.data.validation",
        "src.detection.mobilenetv3",
        "src.detection.classifier_train",
        "src.detection.classifier_evaluate",
        "src.detection.predict",
        "src.app.main_window",
        "scripts.download_datasets",
    ],
)
def test_public_modules_import(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module_name",
    [
        "src.data.dedup",
        "src.detection.classifier_evaluate",
        "src.data.run_pipeline",
        "src.detection.classifier_train",
        "src.app.main_window",
    ],
)
def test_public_modules_execute_workspace_bytecode(module_name: str) -> None:
    """Reject copied bytecode whose tracebacks point at another checkout."""
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    assert module_path.is_relative_to(PROJECT_ROOT)

    local_code_objects = [
        function
        for _, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module_name
        and Path(function.__code__.co_filename).name == module_path.name
    ]
    for _, class_ in inspect.getmembers(module, inspect.isclass):
        if class_.__module__ != module_name:
            continue
        local_code_objects.extend(
            method
            for _, method in inspect.getmembers(class_, inspect.isfunction)
            if method.__module__ == module_name
            and Path(method.__code__.co_filename).name == module_path.name
        )

    assert local_code_objects, f"Expected executable code in {module_name}"
    for function in local_code_objects:
        code_path = Path(function.__code__.co_filename).resolve()
        assert code_path.is_relative_to(PROJECT_ROOT), (
            f"{module_name}.{function.__name__} executes bytecode compiled from {code_path}"
        )
