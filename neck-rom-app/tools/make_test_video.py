"""Make a synthetic test video from portrait.jpg by rotating it in the image plane.

Two modes:
  scripted (default): timed to the guided flow started with --auto. The image is
      still during calibration, rotated CLOCKWISE (top of head -> image right =
      subject's LEFT) during the "tilt_left" measurement and COUNTER-clockwise
      during "tilt_right". If the roll sign is correct, both tilts read ~amp;
      if the sign were wrong both would read ~0.
  sine: plain back-and-forth roll of +-amp degrees.

Because the WHOLE image rotates, the shoulders rotate too. With Pose enabled the
app should therefore report rom_head ~ amp, rom_neck ~ 0 and compensation=True
for the tilts (this also checks the trunk_roll sign). Use --no-pose in the app
to see only the head angle.

    python tools/make_test_video.py                 # -> test.mp4
    python run.py --video test.mp4 --auto --headless --save-screens out/ --user test
"""
from __future__ import annotations

import argparse
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from neck_rom import session as S  # noqa: E402

PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"
ASSET = Path(__file__).resolve().parent / "assets" / "portrait.jpg"
AUTO_START = 1.0     # app --auto starts the flow at t = 1 s


def get_portrait(path: Path) -> np.ndarray:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {PORTRAIT_URL}")
        urllib.request.urlretrieve(PORTRAIT_URL, path)
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read image {path}")
    return img


def smooth_window(t, t0, t1, ramp=1.0):
    """0 outside [t0, t1], ramps to 1 inside (cosine ramps)."""
    if t <= t0 or t >= t1:
        return 0.0
    a = min(1.0, (t - t0) / ramp, (t1 - t) / ramp)
    return 0.5 - 0.5 * math.cos(math.pi * a)


def scripted_angle(t, amp):
    """cv2 rotation angle (deg, + = counter-clockwise on screen) at time t."""
    t_mov0 = AUTO_START + S.T_CALIB_PREP + S.T_CALIB
    period = S.T_PREP + S.T_MEASURE + S.T_RETURN
    for i, mv in enumerate(S.MOVEMENTS):
        if mv.key not in ("tilt_left", "tilt_right"):
            continue
        m0 = t_mov0 + i * period + S.T_PREP
        w = smooth_window(t, m0 + 0.5, m0 + S.T_MEASURE - 0.3, ramp=1.5)
        if w:
            # tilt_left: top of head to image right (clockwise) -> negative cv2 angle
            return (-amp if mv.key == "tilt_left" else amp) * w
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "test.mp4"))
    ap.add_argument("--image", default=str(ASSET))
    ap.add_argument("--mode", choices=("scripted", "sine"), default="scripted")
    ap.add_argument("--amp", type=float, default=25.0, help="rotation amplitude (deg)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--crop", type=float, default=0.5,
                    help="keep this top fraction of the image (head + shoulders)")
    args = ap.parse_args()

    src = get_portrait(Path(args.image))
    # keep head + shoulders only (like sitting in front of a laptop) so the face is large
    src = src[: int(src.shape[0] * args.crop)]
    W, H = 1280, 720
    s = 0.95 * H / src.shape[0]
    face = cv2.resize(src, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    base = np.full((H, W, 3), 128, np.uint8)
    fh, fw = face.shape[:2]
    x0, y0 = (W - fw) // 2, H - fh
    if fw > W:   # very wide image: crop centre
        face = face[:, (fw - W) // 2:(fw - W) // 2 + W]
        fh, fw = face.shape[:2]
        x0 = 0
    base[y0:y0 + fh, x0:x0 + fw] = face
    center = (W / 2, y0 + fh * 0.45)       # around the neck

    total = AUTO_START + S.T_CALIB_PREP + S.T_CALIB + \
        len(S.MOVEMENTS) * (S.T_PREP + S.T_MEASURE + S.T_RETURN) + 3.0
    seconds = args.seconds or total
    n = int(seconds * args.fps)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not vw.isOpened():
        raise SystemExit(f"Cannot write {args.out}")
    for k in range(n):
        t = k / args.fps
        if args.mode == "scripted":
            ang = scripted_angle(t, args.amp)
        else:
            ang = args.amp * math.sin(2 * math.pi * t / 6.0) if t > 7 else 0.0
        M = cv2.getRotationMatrix2D(center, ang, 1.0)
        vw.write(cv2.warpAffine(base, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    vw.release()
    print(f"Wrote {args.out}: {n} frames, {seconds:.1f} s, mode={args.mode}, amp={args.amp}")


if __name__ == "__main__":
    main()
