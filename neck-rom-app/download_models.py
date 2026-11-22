"""Download the MediaPipe model files into models/ (skips files that already exist).

Usage:  python download_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODELS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _progress(name: str):
    last = [-1]

    def hook(blocks: int, block_size: int, total: int):
        done = blocks * block_size
        step = int(10 * done / total) if total > 0 else int(done / 1e6)
        if step == last[0]:
            return
        last[0] = step
        if total > 0:
            pct = min(100.0, 100.0 * done / total)
            sys.stdout.write(f"\r  {name}: {pct:5.1f}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
        else:
            sys.stdout.write(f"\r  {name}: {done / 1e6:.1f} MB")
        sys.stdout.flush()
    return hook


def download(force: bool = False) -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, url in MODELS.items():
        dest = MODEL_DIR / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"  {name}: already present ({dest.stat().st_size / 1e6:.1f} MB), skipped")
            continue
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp, reporthook=_progress(name))
            tmp.replace(dest)
            print(f"\n  {name}: OK -> {dest}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            if tmp.exists():
                tmp.unlink()
            print(f"\n  {name}: FAILED ({e})\n  URL: {url}")
    return failures


if __name__ == "__main__":
    print(f"Downloading MediaPipe models into {MODEL_DIR}")
    sys.exit(1 if download(force="--force" in sys.argv) else 0)
