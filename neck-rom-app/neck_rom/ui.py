"""Drawing with OpenCV + Pillow (OpenCV putText cannot draw Chinese characters).

Design rules (older adults): 1280x720 canvas, main instructions >= 52 px,
angle numbers >= 64 px, secondary text >= 26 px, disclaimer always visible.

All colours are BGR. The Pillow text layer draws straight onto the BGR buffer
(Pillow thinks it is RGB), so BGR colour tuples can be passed to it unchanged
and no colour conversion is needed per frame.

Usage per frame:
    draw_axes / draw_shoulders on the UN-mirrored camera frame
    -> flip (mirror) -> fit_canvas -> ui.render_live(...)
Text is queued with ui.text()/ui.label() and drawn in one pass by ui.flush().
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .i18n import T
from .session import COMP_THRESHOLD_DEG, MOVEMENTS

W, H = 1280, 720

# ---- colours (BGR) ---------------------------------------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT = (215, 215, 215)
GREY = (150, 150, 150)
DARK = (70, 70, 70)
BG = (38, 32, 28)
CELL_BG = (58, 50, 45)
GREEN = (90, 200, 80)
YELLOW = (0, 215, 255)
RED = (60, 60, 230)
ORANGE = (0, 150, 255)
BLUE = (255, 140, 30)
ARROW = (0, 230, 255)
BTN = (110, 75, 40)          # default button (blue-ish)
BTN_GO = (70, 150, 60)       # start / next (green)
BTN_DIM = (90, 90, 90)       # skip (grey)

FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
LATIN_FALLBACK = [
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Chinese punctuation that must not start a new line
_NO_LINE_START = set("，。！？、）」』：；…,.!?:;)")


def find_font(override: str | None = None) -> tuple[str | None, bool]:
    """Return (font_path, has_cjk)."""
    if override:
        if Path(override).exists():
            return override, True
        print(f"[ui] WARNING: font not found: {override}")
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p, True
    for p in LATIN_FALLBACK:
        if Path(p).exists():
            return p, False
    return None, False


def color_for(rom: float | None, ref: float) -> tuple[int, int, int]:
    if rom is None:
        return GREY
    pct = rom / ref if ref else 0.0
    if pct >= 0.75:
        return GREEN
    if pct >= 0.5:
        return YELLOW
    return RED


# ---- drawing on the raw camera frame (call BEFORE mirroring) ---------------
def draw_axes(frame: np.ndarray, R: np.ndarray, nose_px) -> None:
    """Head axes at the nose: red = r (to subject's left), green = u (up),
    blue = f (out of the face). Image coords: x = nx + L*X, y = ny - L*Y."""
    if R is None or nose_px is None:
        return
    nx, ny = nose_px
    L = 0.12 * frame.shape[1]
    for c, col in ((2, BLUE), (0, RED), (1, GREEN)):
        end = (int(nx + L * R[0, c]), int(ny - L * R[1, c]))
        cv2.arrowedLine(frame, (nx, ny), end, col, 3, cv2.LINE_AA, tipLength=0.18)


def draw_shoulders(frame: np.ndarray, shoulders_px, warn: bool) -> None:
    if not shoulders_px:
        return
    (lx, ly), (rx, ry) = shoulders_px
    col = RED if warn else GREEN
    cv2.line(frame, (lx, ly), (rx, ry), col, 6, cv2.LINE_AA)
    for p in ((lx, ly), (rx, ry)):
        cv2.circle(frame, p, 10, col, -1, cv2.LINE_AA)


def fit_canvas(frame: np.ndarray) -> np.ndarray:
    """Resize (keeping aspect ratio) and letterbox into a W x H canvas."""
    h, w = frame.shape[:2]
    if (w, h) == (W, H):
        return frame
    s = min(W / w, H / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((H, W, 3), np.uint8)
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _arrow_head(img, tip, d, size, color):
    d = np.asarray(d, float)
    d /= np.linalg.norm(d) + 1e-9
    n = np.array([-d[1], d[0]])
    tip = np.asarray(tip, float)
    base = tip - d * size
    pts = np.array([tip, base + n * size * 0.65, base - n * size * 0.65], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


class UI:
    def __init__(self, t: T, font_path: str | None = None, mirror: bool = True):
        self.t = t
        self.mirror = mirror
        self.font_path, self.cjk = find_font(font_path)
        if not self.cjk:
            print("[ui] WARNING: no Chinese (CJK) font found, Chinese text will not render.\n"
                  "     Use --font <path-to-ttc/ttf> or run with --lang en")
        self._fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self._queue: list[tuple] = []
        # clickable buttons of the last rendered screen: (id, x0, y0, x1, y1), canvas coords
        self.buttons: list[tuple[str, int, int, int, int]] = []

    # ---- text primitives ---------------------------------------------------
    def font(self, size: int):
        f = self._fonts.get(size)
        if f is None:
            if self.font_path:
                f = ImageFont.truetype(self.font_path, size)
            else:
                try:
                    f = ImageFont.load_default(size=size)
                except TypeError:  # very old Pillow
                    f = ImageFont.load_default()
            self._fonts[size] = f
        return f

    def text_w(self, s: str, size: int) -> float:
        return self.font(size).getlength(s)

    def text(self, xy, s: str, size: int, color=WHITE, anchor: str = "la", stroke: int = 0):
        """Queue a text item (drawn by flush)."""
        self._queue.append((tuple(int(v) for v in xy), s, size, color, anchor, stroke))

    def flush(self, img: np.ndarray) -> np.ndarray:
        """Draw all queued text in ONE BGR->PIL->BGR round trip."""
        if not self._queue:
            return img
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for xy, s, size, color, anchor, stroke in self._queue:
            kw = dict(font=self.font(size), fill=color)
            if stroke:
                kw.update(stroke_width=stroke, stroke_fill=BLACK)
            try:
                d.text(xy, s, anchor=anchor, **kw)
            except ValueError:  # bitmap fallback font does not support anchors
                d.text(xy, s, **kw)
        self._queue.clear()
        img[:] = np.asarray(pil)
        return img

    def wrap(self, s: str, size: int, max_w: float) -> list[str]:
        f = self.font(size)
        lines, cur = [], ""
        for ch in s:
            trial = cur + ch
            if cur and ch not in _NO_LINE_START and f.getlength(trial) > max_w:
                cut = cur.rfind(" ")
                if ch != " " and cut > 0 and ord(ch) < 0x2E80:   # latin: break at space
                    lines.append(cur[:cut])
                    cur = cur[cut + 1:] + ch
                else:
                    lines.append(cur.rstrip())
                    cur = ch.lstrip()
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def panel(img, x0, y0, x1, y1, color=BLACK, alpha=0.55):
        """Darken / tint a rectangle (cheaper than RGBA compositing)."""
        hh, ww = img.shape[:2]
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(ww, int(x1)), min(hh, int(y1))
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        overlay = np.empty_like(roi)
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)

    def label(self, img, s: str, size: int, y: int, x: int | None = None, color=WHITE,
              bg=BLACK, alpha=0.55, pad: int = 12, max_w: int = W - 100,
              align: str = "center", stroke: int = 0) -> int:
        """Text (wrapped) on a translucent box. Returns the bottom y of the box."""
        lines = self.wrap(s, size, max_w - 2 * pad)
        lh = int(size * 1.3)
        bw = int(max(self.text_w(ln, size) for ln in lines)) + 2 * pad
        bh = lh * (len(lines) - 1) + int(size * 1.15) + 2 * pad
        if align == "center":
            cx = W // 2 if x is None else x
            x0 = cx - bw // 2
        elif align == "left":
            x0 = 0 if x is None else x
        else:
            x0 = (W if x is None else x) - bw
        if bg is not None:
            self.panel(img, x0, y, x0 + bw, y + bh, bg, alpha)
        for i, ln in enumerate(lines):
            ly = y + pad + i * lh + int(size * 0.55)
            if align == "center":
                self.text((x0 + bw / 2, ly), ln, size, color, "mm", stroke)
            elif align == "left":
                self.text((x0 + pad, ly), ln, size, color, "lm", stroke)
            else:
                self.text((x0 + bw - pad, ly), ln, size, color, "rm", stroke)
        return y + bh

    def fit_size(self, s: str, size: int, max_w: float, min_size: int = 36) -> int:
        """Largest font size <= size so that s fits in max_w (for long English titles)."""
        while size > min_size and self.text_w(s, size) > max_w:
            size -= 2
        return size

    # ---- buttons -------------------------------------------------------------
    def button(self, img, bid: str, x0, y0, x1, y1, label: str, size: int = 30, color=BTN):
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        self.panel(img, x0, y0, x1, y1, color, 0.85)
        cv2.rectangle(img, (x0, y0), (x1, y1), WHITE, 2, cv2.LINE_AA)
        self.text(((x0 + x1) // 2, (y0 + y1) // 2), label, size, WHITE, "mm")
        self.buttons.append((bid, x0, y0, x1, y1))

    def buttons_right(self, img, specs, y0: int = 10, h: int = 48, right: int = W - 14,
                      gap: int = 10, size: int = 28, min_w: int = 96) -> int:
        """Lay out buttons right-to-left. specs = [(id, label, color), ...] in visual
        left-to-right order. Returns the x of the leftmost button."""
        x1 = right
        for bid, label, color in reversed(specs):
            w = max(min_w, int(self.text_w(label, size)) + 40)
            self.button(img, bid, x1 - w, y0, x1, y0 + h, label, size, color)
            x1 -= w + gap
        return x1 + gap

    def hit(self, x: int, y: int) -> str | None:
        """Id of the button under canvas point (x, y), or None."""
        for bid, x0, y0, x1, y1 in reversed(self.buttons):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return bid
        return None

    # ---- shared widgets ------------------------------------------------------
    def _disclaimer(self, img):
        self.panel(img, 0, H - 44, W, H, BLACK, 0.75)
        self.text((W // 2, H - 22), self.t("disclaimer"), 26, YELLOW, "mm")

    def _readout(self, img, sample):
        """Small current angles in the top-right corner (left of the language button)."""
        x = W - 128
        head = sample.head if (sample is not None and sample.valid) else None
        rows = []
        if head:
            rows = [f"yaw {head['yaw']:+6.1f}°", f"pitch {head['pitch']:+6.1f}°",
                    f"roll {head['roll']:+6.1f}°"]
        if sample is not None and sample.trunk:
            tk = sample.trunk
            rows.append(f"{self.t('trunk')} {tk['yaw']:+.0f}° / {tk['roll']:+.0f}°")
        for i, r in enumerate(rows):
            self.text((x, 8 + i * 26), r, 22, LIGHT, "ra")

    def _countdown(self, img, remaining: float | None, y: int = 370):
        if remaining is None:
            return
        n = max(1, math.ceil(remaining - 1e-6))
        self.text((W // 2, y), str(n), 200, WHITE, "mm", stroke=8)

    def _progress(self, img, frac: float, y: int, h: int = 16, color=GREEN):
        frac = min(1.0, max(0.0, frac))
        cv2.rectangle(img, (0, y), (W, y + h), DARK, -1)
        cv2.rectangle(img, (0, y), (int(W * frac), y + h), color, -1)

    def _direction(self, key: str) -> str:
        """Direction to show on the (possibly mirrored) screen."""
        user = {"flexion": "down", "extension": "up", "rot_left": "left", "rot_right": "right",
                "tilt_left": "ccw", "tilt_right": "cw"}[key]
        if not self.mirror:   # un-mirrored view: subject's left is on screen right
            user = {"left": "right", "right": "left", "ccw": "cw", "cw": "ccw"}.get(user, user)
        return user

    def _arrow(self, img, key: str):
        d = self._direction(key)
        # horizontal arrows sit on the side they point to; others on the right
        if d == "left":
            x0, x1 = 20, 200
        else:
            x0, x1 = W - 200, W - 20
        y0, y1 = 250, 450
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        self.panel(img, x0, y0, x1, y1, BLACK, 0.5)
        th = 16
        if d in ("left", "right"):
            s = -1 if d == "left" else 1
            a, b = (cx - s * 65, cy), (cx + s * 45, cy)
            cv2.line(img, a, b, ARROW, th, cv2.LINE_AA)
            _arrow_head(img, (cx + s * 80, cy), (s, 0), 48, ARROW)
        elif d in ("up", "down"):
            s = -1 if d == "up" else 1
            a, b = (cx, cy - s * 65), (cx, cy + s * 45)
            cv2.line(img, a, b, ARROW, th, cv2.LINE_AA)
            _arrow_head(img, (cx, cy + s * 80), (0, s), 48, ARROW)
        else:
            hc = (cx, cy + 15)
            cv2.circle(img, hc, 30, LIGHT, -1, cv2.LINE_AA)       # schematic head
            r = 66
            if d == "ccw":   # head top moves to screen-left
                cv2.ellipse(img, hc, (r, r), 0, 205, 295, ARROW, 12, cv2.LINE_AA)
                th_ = math.radians(205)
                tang = (math.sin(th_), -math.cos(th_))
            else:
                cv2.ellipse(img, hc, (r, r), 0, 245, 335, ARROW, 12, cv2.LINE_AA)
                th_ = math.radians(335)
                tang = (-math.sin(th_), math.cos(th_))
            tip = (hc[0] + r * math.cos(th_) + 22 * tang[0], hc[1] + r * math.sin(th_) + 22 * tang[1])
            _arrow_head(img, tip, tang, 42, ARROW)

    # ---- screens -------------------------------------------------------------
    def render_live(self, img: np.ndarray, flow, sample, t: float, measurement=None,
                    use_pose: bool = True) -> np.ndarray:
        """Home / calibration / prep / measure / return screens (on the camera image)."""
        tr = self.t
        ph, mv, rec = flow.phase, flow.movement, flow.recorder
        dur = flow.duration()
        el = flow.elapsed(t)
        remaining = max(0.0, dur - el) if dur else None
        frac = min(1.0, el / dur) if dur else 0.0

        self.buttons = []
        # top bar
        self.panel(img, 0, 0, W, 112, BLACK, 0.6)
        if ph in ("prep", "measure", "return") and mv:
            self.text((30, 8), tr("step", i=flow.idx + 1, n=len(MOVEMENTS)), 30, LIGHT)
            name = tr(mv.key)
            self.text((30, 46), name, self.fit_size(name, 52, W - 30 - 340), WHITE)
        else:
            self.text((30, 56), tr("app_title"), 56, WHITE, "lm")
        self._readout(img, sample)
        self.buttons_right(img, [("lang", tr("btn_lang"), BTN)])

        # clickable controls (same actions as SPACE / S / H)
        if ph == "home":
            self.button(img, "start", W - 260, 236, W - 20, 326, tr("btn_start"), 48, BTN_GO)
            self.button(img, "history", W - 260, 336, W - 20, 388, tr("btn_history"), 28)
        elif ph == "calib_prep":
            self.button(img, "next", W - 180, 596, W - 16, 660, tr("btn_next"), 32, BTN_GO)
        elif ph in ("prep", "measure", "return"):
            self.button(img, "next", W - 180, 596, W - 16, 660, tr("btn_next"), 32, BTN_GO)
            self.button(img, "skip", W - 320, 596, W - 192, 660, tr("btn_skip"), 30, BTN_DIM)

        warn_shown = False
        if ph == "home":
            y = self.label(img, tr("home_hint"), 52, y=400, max_w=W - 120, alpha=0.6)
            has_trunk = use_pose and measurement is not None and measurement.trunk is not None
            y = self.label(img, tr("comp_on") if has_trunk else tr("comp_off"), 28, y=y + 10,
                           color=GREEN if has_trunk else ORANGE, pad=6, alpha=0.6)
            self.label(img, tr("home_keys"), 28, y=max(y + 10, H - 44 - 62), color=LIGHT,
                       pad=6, alpha=0.6)
        elif ph == "calib_prep":
            y = self.label(img, tr("calib_prep"), 52, y=128)
            if flow.message:
                self.label(img, tr(flow.message), 34, y=y + 10, color=YELLOW)
            self._countdown(img, remaining)
        elif ph == "calib":
            self._progress(img, frac, 112, color=YELLOW)
            self.label(img, tr("calib"), 52, y=140)
        elif ph == "prep" and mv:
            self.label(img, tr("get_ready"), 52, y=128)
            self._countdown(img, remaining)
            self._arrow(img, mv.key)
            self.label(img, tr("safety"), 30, y=540, color=YELLOW)
        elif ph == "measure" and mv:
            self._progress(img, frac, 112)
            self.label(img, tr("measuring"), 52, y=138)
            self._arrow(img, mv.key)
            cur = rec.current if rec else None
            mx = rec.res.rom_head if rec else None
            self.panel(img, 240, 450, 1040, 580, BLACK, 0.6)
            f = lambda v: "--" if v is None else f"{max(0.0, v):.0f}°"  # noqa: E731
            self.text((440, 515), f"{tr('now')} {f(cur)}", 72, WHITE, "mm")
            self.text((840, 515), f"{tr('max')} {f(mx)}", 72, color_for(mx, mv.ref), "mm")
            if rec and rec.trunk_now is not None and rec.current_neck is not None:
                self.label(img, f"{tr('trunk')} {rec.trunk_now:+.0f}°　　{tr('neck')} "
                           f"{max(0.0, rec.current_neck):.0f}°", 28, y=584, color=LIGHT, pad=5)
            if rec and rec.trunk_now is not None and rec.trunk_now > COMP_THRESHOLD_DEG:
                self.label(img, tr("keep_shoulders"), 64, y=280, bg=RED, alpha=0.8, pad=16)
                warn_shown = True
            elif rec and rec.wrong_dir:
                self.label(img, tr("wrong_dir"), 60, y=280, bg=ORANGE, alpha=0.8, pad=16)
                warn_shown = True
        elif ph == "return" and mv:
            self.label(img, tr("return"), 52, y=128)
            last = flow.results[-1] if flow.results else None
            if last is not None and last.key == mv.key and not last.skipped:
                v = last.rom
                txt = f"{tr('max')} {v:.0f}°" if v is not None else tr("no_data")
                self.label(img, txt, 72, y=470, color=color_for(v, mv.ref), alpha=0.6)

        if sample is not None and not warn_shown:
            msg = sample.reason if not sample.valid else getattr(sample, "hint", None)
            if msg:
                self.label(img, tr(msg), 44, y=290, color=YELLOW, alpha=0.7)

        if ph in ("prep", "measure", "return"):
            self.text((W // 2, H - 44 - 22), tr("space_next"), 26, LIGHT, "mm", stroke=2)
        self._disclaimer(img)
        return self.flush(img)

    def render_result(self, results, saved: str | None = None) -> np.ndarray:
        """Horizontal bars: grey box = reference, coloured bar = (compensated) ROM."""
        tr = self.t
        img = np.full((H, W, 3), BG, np.uint8)
        self.buttons = []
        self.text((40, 46), tr("result_title"), 56, WHITE, "lm")
        self.buttons_right(img, [("history", tr("btn_history"), BTN),
                                 ("home", tr("btn_home"), BTN),
                                 ("again", tr("btn_again"), BTN_GO),
                                 ("lang", tr("btn_lang"), BTN)])
        self.text((W - 14, 78), tr("result_keys"), 22, GREY, "rm")
        if saved:
            self.text((40, 96), tr("saved", p=saved), 20, GREY, "lm")

        by_key = {(r.key if hasattr(r, "key") else r["key"]): r for r in results}
        vals = [mv.ref for mv in MOVEMENTS]
        for r in by_key.values():
            for k in ("rom", "rom_head"):
                v = _get(r, k)
                if v is not None:
                    vals.append(v)
        vmax = max(100.0, max(vals) + 5)
        bx0, bx1 = 270, 1060
        scale = (bx1 - bx0) / vmax
        y0, row_h = 108, 88
        for i, mv in enumerate(MOVEMENTS):
            y = y0 + i * row_h
            r = by_key.get(mv.key)
            by, bh = y + 4, 46
            self.text((40, by + bh // 2), tr(mv.key + "_s"), 40, WHITE, "lm")
            xr = bx0 + int(mv.ref * scale)
            cv2.rectangle(img, (bx0, by), (xr, by + bh), (62, 58, 56), -1)
            rom = _get(r, "rom") if r is not None else None
            if r is None or _get(r, "skipped") or rom is None:
                cv2.rectangle(img, (bx0, by), (xr, by + bh), GREY, 3)
                msg = tr("skipped") if (r is not None and _get(r, "skipped")) else tr("no_data")
                self.text((bx0 + 20, by + bh // 2), msg, 34, GREY, "lm")
                continue
            col = color_for(rom, mv.ref)
            cv2.rectangle(img, (bx0, by + 6), (bx0 + int(rom * scale), by + bh - 6), col, -1)
            cv2.rectangle(img, (bx0, by), (xr, by + bh), GREY, 3)
            self.text((bx1 + 20, by + bh // 2), f"{rom:.0f}°", 56, col, "lm")
            rom_head = _get(r, "rom_head")
            if _get(r, "compensation") and rom_head is not None:
                xh = bx0 + int(rom_head * scale)
                cv2.line(img, (xh, by - 4), (xh, by + bh + 4), WHITE, 3, cv2.LINE_AA)
                mt = _get(r, "max_trunk")
                self.text((bx0, by + bh + 4), tr("comp_detected", v=f"{(mt or 0):.0f}",
                                                 raw=f"{rom_head:.0f}"), 24, ORANGE)

        # legend
        ly = 650
        items = [(GREEN, "legend_ok"), (YELLOW, "legend_mid"), (RED, "legend_low"), (None, "ref_label")]
        for fs in (26, 24, 22, 20):      # shrink until the legend fits on one line
            gap = 36 if fs == 26 else 24
            total = sum(44 + self.text_w(tr(k), fs) + gap for _, k in items) - gap
            if total <= W - 60:
                break
        x = 40
        for col, key in items:
            if col is None:
                cv2.rectangle(img, (x, ly - 12), (x + 34, ly + 12), GREY, 3)
            else:
                cv2.rectangle(img, (x, ly - 12), (x + 34, ly + 12), col, -1)
            self.text((x + 44, ly), tr(key), fs, LIGHT, "lm")
            x += 44 + int(self.text_w(tr(key), fs)) + gap
        self._disclaimer(img)
        return self.flush(img)

    def render_history(self, sessions: list[dict]) -> np.ndarray:
        """2 x 3 grid of small line charts, one per movement."""
        tr = self.t
        img = np.full((H, W, 3), BG, np.uint8)
        self.buttons = []      # the whole history screen acts as a "back" button
        self.text((40, 44), tr("history_title"), 44, WHITE, "lm")
        self.text((W - 30, 44), tr("history_keys"), 26, LIGHT, "rm")
        if not sessions:
            self.text((W // 2, H // 2), tr("history_empty"), 56, GREY, "mm")
            self._disclaimer(img)
            return self.flush(img)

        dates = []
        for s in sessions:
            try:
                dates.append(datetime.fromisoformat(s.get("started_at", "")).strftime("%m/%d"))
            except ValueError:
                dates.append("?")
        n = len(sessions)
        gx0, gy0, gx1, gy1 = 24, 88, W - 24, H - 56
        cw, ch = (gx1 - gx0) / 3, (gy1 - gy0) / 2
        for i, mv in enumerate(MOVEMENTS):
            cx0 = int(gx0 + (i % 3) * cw) + 6
            cy0 = int(gy0 + (i // 3) * ch) + 6
            cx1, cy1 = int(cx0 + cw) - 12, int(cy0 + ch) - 12
            cv2.rectangle(img, (cx0, cy0), (cx1, cy1), CELL_BG, -1)
            self.text((cx0 + 14, cy0 + 22), tr(mv.key + "_s"), 30, WHITE, "lm")
            vals = []
            for s in sessions:
                r = next((r for r in s.get("results", []) if r.get("key") == mv.key), None)
                vals.append(r.get("rom") if r and not r.get("skipped") else None)
            px0, px1 = cx0 + 52, cx1 - 18
            py0, py1 = cy0 + 60, cy1 - 34
            good = [v for v in vals if v is not None]
            vmax = max([100.0, mv.ref + 10] + [v + 10 for v in good])
            vmax = math.ceil(vmax / 20) * 20
            ypix = lambda v: int(py1 - (py1 - py0) * v / vmax)  # noqa: E731
            for tick in range(0, int(vmax) + 1, 20 if vmax <= 120 else 40):
                yy = ypix(tick)
                cv2.line(img, (px0, yy), (px1, yy), (75, 68, 62), 1)
                self.text((px0 - 8, yy), str(tick), 20, GREY, "rm")
            # reference: dashed line
            yr = ypix(mv.ref)
            for xx in range(px0, px1, 16):
                cv2.line(img, (xx, yr), (min(xx + 8, px1), yr), LIGHT, 2)
            xpix = lambda k: int(px0 + (k + 0.5) * (px1 - px0) / n)  # noqa: E731
            pts = [(xpix(k), ypix(v)) for k, v in enumerate(vals) if v is not None]
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, np.int32)], False, WHITE, 3, cv2.LINE_AA)
            for k, v in enumerate(vals):
                if v is not None:
                    cv2.circle(img, (xpix(k), ypix(v)), 7, color_for(v, mv.ref), -1, cv2.LINE_AA)
            if good:
                last = next(v for v in reversed(vals) if v is not None)
                self.text((cx1 - 14, cy0 + 22), f"{last:.0f}°", 30, color_for(last, mv.ref), "rm")
            step = max(1, math.ceil(n / 6))
            for k in range(0, n, step):
                self.text((xpix(k), py1 + 16), dates[k], 20, GREY, "mm")
        self._disclaimer(img)
        return self.flush(img)


def _get(r, k):
    """Field access for both MovementResult objects and dicts loaded from JSON."""
    if r is None:
        return None
    if isinstance(r, dict):
        return r.get(k)
    return getattr(r, k, None)
