"""Main loop: camera / video -> Tracker -> Flow -> UI.

    python run.py                          # webcam 0
    python run.py --video test.mp4 --auto --headless --save-screens out/
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2

from . import storage
from . import ui as U
from .i18n import T
from .session import COMP_THRESHOLD_DEG, AngleProcessor, Flow
from .tracker import Tracker

WINDOW = "Neck ROM"   # ASCII: OpenCV window titles do not support Chinese on Windows
ROOT = Path(__file__).resolve().parent.parent
LIVE_PHASES = ("prep", "measure", "return")
# on-screen button id -> the keyboard key it stands for
BUTTON_KEYS = {"lang": "l", "start": " ", "next": " ", "skip": "s", "history": "h",
               "again": "r", "home": " "}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Webcam neck mobility (CROM) check")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    src.add_argument("--video", type=str, default=None, help="video file instead of webcam")
    p.add_argument("--user", default="default", help="person name, results go to data/<user>/")
    p.add_argument("--lang", choices=("zh", "en"), default="zh")
    p.add_argument("--font", default=None, help="path to a CJK .ttc/.ttf font")
    p.add_argument("--no-pose", action="store_true", help="disable trunk compensation (no Pose model)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the display")
    p.add_argument("--auto", action="store_true", help="start the measurement automatically after 1 s")
    p.add_argument("--headless", action="store_true", help="no window (automatic tests)")
    p.add_argument("--save-screens", metavar="DIR", default=None,
                   help="save one screenshot per step + the result screen (for slides)")
    p.add_argument("--log-csv", metavar="PATH", default=None,
                   help="per-frame angle log (for IMU comparison)")
    return p.parse_args(argv)


def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps < 1:
            fps = 30.0
        return cap, fps
    cap = None
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)   # opens much faster on Windows
        if not cap.isOpened():
            cap.release()
            cap = None
    if cap is None:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Try --camera 1, or check permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap, None


class CsvLogger:
    COLS = ["t", "phase", "movement", "valid", "yaw", "pitch", "roll",
            "yaw_raw", "pitch_raw", "roll_raw", "trunk_yaw", "trunk_roll"]

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow(self.COLS)

    def write(self, t, flow, s):
        f = lambda d, k: "" if not d else f"{d[k]:.3f}"  # noqa: E731
        mv = flow.movement.key if (flow.movement and flow.phase in LIVE_PHASES) else ""
        self.w.writerow([f"{t:.4f}", flow.phase, mv, int(s.valid),
                         f(s.head, "yaw"), f(s.head, "pitch"), f(s.head, "roll"),
                         f(s.head_raw, "yaw"), f(s.head_raw, "pitch"), f(s.head_raw, "roll"),
                         f(s.trunk, "yaw"), f(s.trunk, "roll")])

    def close(self):
        self.f.close()


def imwrite(path: Path, img) -> None:
    """cv2.imwrite fails on non-ASCII paths on Windows; encode + tofile works everywhere."""
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if ok:
        buf.tofile(str(path))


def shot_key(flow, now, screen):
    """When to grab a screenshot: once per phase, 60% into timed phases."""
    if screen == "history":
        return ("history", 0)
    ph = flow.phase
    if ph in ("home", "result"):
        return (ph, flow.idx)
    dur = flow.duration()
    if dur and flow.elapsed(now) >= 0.6 * dur:
        return (ph, flow.idx)
    return None


def print_summary(flow):
    print("\n=== Results ===")
    print(f"{'movement':<11}{'rom':>7}{'head':>7}{'raw':>7}{'neck':>7}{'trunk':>7}"
          f"  comp  {'frames':<20}rejected")
    fmt = lambda v: "   -  " if v is None else f"{v:6.1f} "  # noqa: E731
    for r in flow.results:
        if r.skipped:
            print(f"{r.key:<11}  (skipped)")
            continue
        frames = f"{r.valid_frames}/{r.total_frames} (trunk {r.trunk_frames})"
        rejected = " ".join(f"{k.removeprefix('q_')}:{n}"
                            for k, n in sorted(r.rejected.items())) or "-"
        print(f"{r.key:<11}{fmt(r.rom)}{fmt(r.rom_head)}{fmt(r.rom_head_raw)}{fmt(r.rom_neck)}"
              f"{fmt(r.max_trunk)}  {'YES ' if r.compensation else 'no  '}  "
              f"{frames:<20}{rejected}")


def _rel(p: Path) -> str:
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def main(argv=None) -> int:
    args = parse_args(argv)
    tr = T(args.lang)
    ui = U.UI(tr, font_path=args.font, mirror=not args.no_mirror)
    use_pose = not args.no_pose
    try:
        tracker = Tracker(use_pose=use_pose)
    except FileNotFoundError as e:
        print(e)
        return 1
    cap, fps = open_capture(args)
    is_video = args.video is not None
    source = f"video:{Path(args.video).name}" if is_video else f"webcam:{args.camera}"

    proc = AngleProcessor()
    flow = Flow(proc)
    screen, hist_canvas = "live", None
    saved_flow, saved_path = None, None
    logger = CsvLogger(args.log_csv) if args.log_csv else None
    shots_dir = Path(args.save_screens) if args.save_screens else None
    if shots_dir:
        shots_dir.mkdir(parents=True, exist_ok=True)
    shots_done, n_shots = set(), 0

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        # OpenCV reports image (canvas) coordinates, also when the window is resized
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, U.W, U.H)
        cv2.setMouseCallback(WINDOW, on_mouse)
    fullscreen = False
    t0 = time.monotonic()
    frame_idx, fails, auto_started = 0, 0, False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_video:
                    break
                fails += 1
                if fails > 50:
                    print("Camera stopped delivering frames.")
                    break
                continue
            fails = 0
            now = frame_idx / fps if is_video else time.monotonic() - t0
            frame_idx += 1

            m = tracker.process(frame, int(now * 1000))
            if args.auto and not auto_started and flow.phase == "home" and now >= 1.0:
                flow.start(now)
                auto_started = True
            s = flow.update(m, now)

            if flow.finished and saved_flow is not flow:
                p = storage.save_session(args.user, flow.to_dict(), source)
                saved_flow, saved_path = flow, _rel(p)
                print_summary(flow)
                print(f"Saved: {p}")
            if logger:
                logger.write(now, flow, s)

            # ---- render ----
            if screen == "history":
                canvas = hist_canvas
            elif flow.phase == "result":
                canvas = ui.render_result(flow.results, saved=saved_path)
            else:
                vis = frame.copy()
                U.draw_axes(vis, m.R, m.nose_px)
                rec = flow.recorder
                warn = bool(rec and rec.trunk_now is not None and rec.trunk_now > COMP_THRESHOLD_DEG)
                U.draw_shoulders(vis, m.shoulders_px, warn)
                if not args.no_mirror:
                    vis = cv2.flip(vis, 1)
                canvas = ui.render_live(U.fit_canvas(vis), flow, s, now, m, use_pose=use_pose)

            if shots_dir:
                k = shot_key(flow, now, screen)
                if k and k not in shots_done:
                    shots_done.add(k)
                    mv = flow.movement.key if (flow.movement and k[0] in LIVE_PHASES) else ""
                    name = f"{n_shots:02d}_{k[0]}{'_' + mv if mv else ''}.png"
                    imwrite(shots_dir / name, canvas)
                    n_shots += 1

            if args.headless:
                if is_video and flow.phase == "result":
                    break
                continue

            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            # keyboard + mouse clicks -> list of key commands
            cmds = []
            if key != 255:
                cmds.append("esc" if key == 27 else (chr(key).lower() if key < 128 else "?"))
            while clicks:
                x, y = clicks.pop(0)
                if screen == "history":
                    cmds.append("back")             # click anywhere: back
                else:
                    bid = ui.hit(x, y)
                    if bid:
                        cmds.append(BUTTON_KEYS[bid])
            quit_app = False
            for ch in cmds:
                if screen == "history":       # any key / click: back
                    screen = "live"
                    continue
                if ch in ("esc", "q"):
                    quit_app = True
                    break
                flow, screen, hist_canvas, fullscreen = handle_key(
                    ch, flow, proc, now, screen, hist_canvas, fullscreen, ui, tr, args)
            if quit_app:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tracker.close()
        if logger:
            logger.close()
        if not args.headless:
            cv2.destroyAllWindows()

    if is_video and not flow.finished:
        print(f"Video ended in phase '{flow.phase}' (movement {flow.idx + 1}).")
        if flow.results:
            print_summary(flow)
    return 0


def handle_key(ch, flow, proc, now, screen, hist_canvas, fullscreen, ui, tr, args):
    """Apply one key command (from the keyboard or an on-screen button).

    Returns the (possibly new) flow, screen, history canvas and fullscreen state.
    """
    if ch == "l":
        tr.toggle()
    elif ch == "f":
        fullscreen = not fullscreen
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    elif ch == " ":
        if flow.phase == "home":
            flow.start(now)
        elif flow.phase == "result":
            flow = Flow(proc)
        else:
            flow.next(now)
    elif ch == "s":
        flow.skip(now)
    elif ch == "c":
        if flow.phase in ("home", "result"):
            flow = Flow(proc)
            flow.start(now)
        else:   # recalibrate, then continue with the current movement
            flow.recorder = None
            flow._start_calib(now)
    elif ch == "r":
        flow = Flow(proc)
        flow.start(now)
    elif ch == "h" and flow.phase in ("home", "result"):
        hist_canvas = ui.render_history(storage.load_sessions(args.user))
        screen = "history"
    return flow, screen, hist_canvas, fullscreen


if __name__ == "__main__":
    sys.exit(main())
