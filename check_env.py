"""Local preflight checks for NeuralVision AI.

Run with the project virtual environment:
    .\venv\Scripts\python.exe check_env.py
"""

import importlib
import shutil
import sys
from pathlib import Path


CRITICAL_MODULES = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("pytorchvideo", "PyTorchVideo"),
    ("cv2", "OpenCV"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("google.genai", "Google GenAI"),
]


def check_module(module_name: str, label: str) -> bool:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "OK")
        print(f"OK   {label}: {version}")
        return True
    except ImportError as exc:
        print(f"MISS {label}: {exc}")
        return False


def main() -> int:
    print("=== NeuralVision AI Local Health Check ===")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    venv_python = Path("venv/Scripts/python.exe")
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        print("WARN You are not running the project venv Python.")
        print(r"     Recommended: .\venv\Scripts\python.exe check_env.py")

    ok = True
    for module_name, label in CRITICAL_MODULES:
        ok = check_module(module_name, label) and ok

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU: none detected; inference will use CPU and may be slow.")
    except ImportError:
        ok = False

    print(f"Node: {shutil.which('node') or 'not found'}")
    print(f"npm:  {shutil.which('npm') or 'not found'}")

    for path in [
        Path("main.py"),
        Path("frontend/package.json"),
        Path("kinetics_classnames.json"),
    ]:
        if path.exists():
            print(f"OK   Found {path}")
        else:
            print(f"MISS Missing {path}")
            ok = False

    print("=== Check complete ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
