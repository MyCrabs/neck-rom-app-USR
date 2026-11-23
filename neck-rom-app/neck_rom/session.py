"""Measurement logic: calibration, smoothing, guided 6-movement flow, ROM.

This module has no UI code so it can be unit-tested without a camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import geometry as g
from .filters import OneEuro

AXES = ("yaw", "pitch", "roll")


@dataclass(frozen=True)
class Movement:
    key: str          # id, also the i18n key
    axis: str         # "yaw" | "pitch" | "roll"
    sign: int         # +1: ROM = theta - theta0 ; -1: ROM = theta0 - theta
    ref: float        # common adult reference value (deg), see README
    comp_axis: str | None  # trunk axis used for compensation (None = not measured)


# Order follows the guided flow. Signs follow spec section 4.3.
MOVEMENTS = [
    Movement("flexion", "pitch", -1, 50.0, None),
    Movement("extension", "pitch", +1, 60.0, None),
    Movement("tilt_left", "roll", +1, 45.0, "roll"),
    Movement("tilt_right", "roll", -1, 45.0, "roll"),
    Movement("rot_left", "yaw", +1, 80.0, "yaw"),
    Movement("rot_right", "yaw", -1, 80.0, "yaw"),
]

# Timings (seconds)
T_CALIB_PREP = 3.0
T_CALIB = 2.0
T_PREP = 3.0
T_MEASURE = 7.0
T_RETURN = 2.5

COMP_THRESHOLD_DEG = 10.0      # trunk moved more than this -> compensation warning
COMP_MIN_FRAMES = 5            # ... for at least this many frames -> compensation detected
# MediaPipe Pose infers shoulder depth partly from the head, so a head-only turn
# makes the shoulder angles drift 15-30 deg. The guards below reject that.
TRUNK_DEADBAND_DEG = 8.0       # trunk angles below this are treated as 0
TRUNK_MAX_FRAC = 0.5           # never subtract more than this fraction of the head angle
WIDTH_TOL = 0.02               # tolerance on the 2D shoulder-width check for rotation
JUMP_REJECT_DEG = 35.0        # one-frame jump larger than this -> outlier frame
CALIB_MAX_STD_DEG = 3.0        # head must be still during calibration
MIN_VALID_FRAMES = 8


@dataclass
class Sample:
    t: float
    valid: bool
    head: dict | None = None        # filtered relative angles {yaw,pitch,roll}
    head_raw: dict | None = None    # unfiltered relative angles
    trunk: dict | None = None       # filtered relative trunk {yaw, roll}
    shoulder_ratio: float | None = None  # 2D shoulder width / width at calibration
    reason: str | None = None       # i18n key when invalid
    hint: str | None = None         # i18n key shown on valid frames (e.g. "q_too_far")


class AngleProcessor:
    """Turns raw per-frame measurements into calibrated, filtered angles."""

    def __init__(self):
        self.R0 = None
        self.trunk0 = None
        self.width0 = None
        self._f = {k: OneEuro(min_cutoff=1.2, beta=0.03) for k in AXES}
        self._ft = {k: OneEuro(min_cutoff=1.0, beta=0.02) for k in ("yaw", "roll")}
        self._fw = OneEuro(min_cutoff=1.0, beta=0.0)
        self._prev = None
        self._rejects = 0

    @property
    def calibrated(self) -> bool:
        return self.R0 is not None

    def reset_filters(self):
        for f in list(self._f.values()) + list(self._ft.values()) + [self._fw]:
            f.reset()
        self._prev, self._rejects = None, 0

    def set_calibration(self, R0, trunk0, width0=None):
        self.R0, self.trunk0, self.width0 = R0, trunk0, width0
        self.reset_filters()

    def update(self, m, t: float) -> Sample:
        if not m.face_found or m.R is None:
            return Sample(t, False, reason=m.quality_msg or "q_no_face")
        # too far is only a hint: the frame is still measured
        hint = "q_too_far" if m.quality_msg == "q_too_far" else None

        if self.R0 is not None:
            y, p, r = g.relative_angles(m.R, self.R0)
        else:
            y, p, r = g.angles_from_matrix(m.R)
        raw = {"yaw": y, "pitch": p, "roll": r}

        # outlier rejection: sudden jumps are almost always landmark failures
        if self._prev is not None and self._rejects < 5:
            jump = max(abs(raw[k] - self._prev[k]) for k in AXES)
            if jump > JUMP_REJECT_DEG:
                self._rejects += 1
                return Sample(t, False, head_raw=raw, reason="q_unstable", hint=hint)
        self._rejects = 0

        head = {k: self._f[k](raw[k], t) for k in AXES}
        self._prev = head

        trunk, ratio = None, None
        if m.trunk is not None:
            ty, tr = m.trunk
            if self.trunk0 is not None:
                ty, tr = g.wrap_deg(ty - self.trunk0[0]), g.wrap_deg(tr - self.trunk0[1])
            trunk = {"yaw": self._ft["yaw"](ty, t), "roll": self._ft["roll"](tr, t)}
            sw = m.shoulder_w   # pixels; only the ratio to calibration is used
            if sw is not None and self.width0:
                ratio = self._fw(sw / self.width0, t)
        return Sample(t, True, head=head, head_raw=raw, trunk=trunk,
                      shoulder_ratio=ratio, hint=hint)


@dataclass
class MovementResult:
    key: str
    rom_head: float | None = None       # max head angle (filtered), = paper's ROM
    rom_head_raw: float | None = None   # max of unfiltered angle (for ablation)
    rom_neck: float | None = None       # max (head - trunk) = compensated neck ROM
    max_trunk: float | None = None      # largest trunk movement actually subtracted
    max_trunk_raw: float | None = None  # largest Pose trunk angle before the guards
    compensation: bool = False
    valid_frames: int = 0
    total_frames: int = 0
    trunk_frames: int = 0
    comp_frames: int = 0                # frames with confirmed trunk > COMP_THRESHOLD_DEG
    rejected: dict = field(default_factory=dict)   # rejection reason -> frame count
    skipped: bool = False

    @property
    def rom(self) -> float | None:
        """Value to report: neck (head - trunk) only if compensation was detected."""
        if self.compensation and self.rom_neck is not None:
            return self.rom_neck
        return self.rom_head

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["rom"] = self.rom
        return d


class MovementRecorder:
    def __init__(self, mv: Movement):
        self.mv = mv
        self.res = MovementResult(mv.key)
        self.current = None      # current directional head angle
        self.current_neck = None
        self.trunk_now = None
        self.wrong_dir = False

    def add(self, s: Sample):
        mv, res = self.mv, self.res
        res.total_frames += 1
        if not s.valid:
            key = s.reason or "unknown"
            res.rejected[key] = res.rejected.get(key, 0) + 1
            return
        res.valid_frames += 1
        head = mv.sign * s.head[mv.axis]
        raw = mv.sign * s.head_raw[mv.axis]
        self.current = head
        self.wrong_dir = head < -12.0
        res.rom_head = head if res.rom_head is None else max(res.rom_head, head)
        res.rom_head_raw = raw if res.rom_head_raw is None else max(res.rom_head_raw, raw)
        self.trunk_now = None
        self.current_neck = None
        if mv.comp_axis and s.trunk is not None:
            raw_trunk = mv.sign * s.trunk[mv.comp_axis]
            trunk = self._confirmed_trunk(raw_trunk, head, s.shoulder_ratio)
            neck = head - trunk
            self.trunk_now, self.current_neck = trunk, neck
            res.trunk_frames += 1
            res.max_trunk_raw = (raw_trunk if res.max_trunk_raw is None
                                 else max(res.max_trunk_raw, raw_trunk))
            res.max_trunk = trunk if res.max_trunk is None else max(res.max_trunk, trunk)
            res.rom_neck = neck if res.rom_neck is None else max(res.rom_neck, neck)
            if trunk > COMP_THRESHOLD_DEG:
                res.comp_frames += 1
                if res.comp_frames >= COMP_MIN_FRAMES:
                    res.compensation = True

    def _confirmed_trunk(self, trunk: float, head: float, ratio: float | None) -> float:
        """Trunk angle that is safe to subtract from the head angle (>= 0)."""
        if trunk < TRUNK_DEADBAND_DEG:
            return 0.0
        if self.mv.comp_axis == "yaw":
            # A real trunk turn also narrows the 2D shoulder line (width ~ cos).
            # Count only as much rotation as that narrowing supports; a width
            # change within WIDTH_TOL counts as none. Using the width-implied
            # angle (not an on/off gate) keeps one lagging frame from dropping
            # the trunk to 0 and making rom_neck jump to the full head angle.
            if ratio is None or ratio > 1.0 - WIDTH_TOL:
                return 0.0
            trunk = min(trunk, math.degrees(math.acos(max(0.0, ratio))))
            if trunk < TRUNK_DEADBAND_DEG:
                return 0.0
        return min(trunk, TRUNK_MAX_FRAC * max(head, 0.0))

    def finish(self) -> MovementResult:
        res = self.res
        if res.valid_frames < MIN_VALID_FRAMES:
            res.rom_head = res.rom_neck = None
        elif res.trunk_frames < 0.5 * res.valid_frames:
            res.rom_neck = None   # not enough shoulder data -> no compensation
        if res.rom_neck is None:
            res.compensation = False
        for k in ("rom_head", "rom_head_raw", "rom_neck", "max_trunk", "max_trunk_raw"):
            v = getattr(res, k)
            if v is not None:
                setattr(res, k, round(float(max(0.0, v) if k.startswith("rom") else v), 1))
        return res


class Flow:
    """Guided measurement state machine.

    phases: home -> calib_prep -> calib -> (prep -> measure -> return) x 6 -> result
    """

    def __init__(self, proc: AngleProcessor):
        self.proc = proc
        self.phase = "home"
        self.t_phase = 0.0
        self.idx = 0
        self.results: list[MovementResult] = []
        self.recorder: MovementRecorder | None = None
        self.message: str | None = None   # transient i18n key
        self._calib_R, self._calib_trunk, self._calib_abs, self._calib_width = [], [], [], []
        self.started_at: datetime | None = None
        self.finished = False

    # ---- helpers -------------------------------------------------------
    @property
    def movement(self) -> Movement | None:
        return MOVEMENTS[self.idx] if self.idx < len(MOVEMENTS) else None

    def elapsed(self, t):
        return t - self.t_phase

    def duration(self):
        return {"calib_prep": T_CALIB_PREP, "calib": T_CALIB, "prep": T_PREP,
                "measure": T_MEASURE, "return": T_RETURN}.get(self.phase)

    def _go(self, phase, t):
        self.phase, self.t_phase = phase, t

    # ---- controls ------------------------------------------------------
    def start(self, t):
        self.idx, self.results, self.finished = 0, [], False
        self.started_at = datetime.now()
        self.message = None
        self._start_calib(t)

    def _start_calib(self, t):
        self._calib_R, self._calib_trunk, self._calib_abs, self._calib_width = [], [], [], []
        self._go("calib_prep", t)

    def next(self, t):
        """SPACE: skip waiting / end the current measurement early."""
        if self.phase in ("prep", "return", "calib_prep"):
            self.t_phase = t - 1e3   # makes the timer expire on next update
        elif self.phase == "measure":
            self._finish_movement(t)

    def skip(self, t):
        if self.phase in ("prep", "measure", "return") and self.movement:
            if self.phase != "return":
                r = MovementResult(self.movement.key, skipped=True)
                self.results.append(r)
            self.recorder = None
            self._advance(t)

    # ---- per-frame update ----------------------------------------------
    def update(self, m, t) -> Sample:
        s = self.proc.update(m, t)
        ph, el = self.phase, self.elapsed(t)

        if ph == "calib_prep" and el >= T_CALIB_PREP:
            self._go("calib", t)
        elif ph == "calib":
            if s.valid and m.R is not None:
                self._calib_R.append(m.R)
                self._calib_abs.append(g.angles_from_matrix(m.R))
                if m.trunk is not None:
                    self._calib_trunk.append(m.trunk)
                    sw = m.shoulder_w
                    if sw:
                        self._calib_width.append(sw)
            if el >= T_CALIB:
                self._finish_calib(t)
        elif ph == "prep" and el >= T_PREP:
            self.recorder = MovementRecorder(self.movement)
            self._go("measure", t)
        elif ph == "measure":
            self.recorder.add(s)
            if el >= T_MEASURE:
                self._finish_movement(t)
        elif ph == "return" and el >= T_RETURN:
            self._advance(t)
        return s

    def _finish_calib(self, t):
        if len(self._calib_R) < MIN_VALID_FRAMES:
            self.message = "msg_calib_noface"
            self._start_calib(t)
            return
        std = np.std(np.array(self._calib_abs), axis=0).max()
        if std > CALIB_MAX_STD_DEG:
            self.message = "msg_calib_still"
            self._start_calib(t)
            return
        R0 = g.average_rotation(self._calib_R)
        trunk0 = None
        if len(self._calib_trunk) >= MIN_VALID_FRAMES // 2:
            trunk0 = tuple(np.median(np.array(self._calib_trunk), axis=0))
        width0 = None
        if len(self._calib_width) >= MIN_VALID_FRAMES // 2:
            width0 = float(np.median(self._calib_width))
        self.proc.set_calibration(R0, trunk0, width0)
        self.message = None
        self._go("prep", t)

    def _finish_movement(self, t):
        self.results.append(self.recorder.finish())
        self.recorder = None
        self._go("return", t)

    def _advance(self, t):
        self.idx += 1
        if self.idx >= len(MOVEMENTS):
            self.finished = True
            self._go("result", t)
        else:
            self._go("prep", t)

    def to_dict(self) -> dict:
        return {
            "started_at": (self.started_at or datetime.now()).isoformat(timespec="seconds"),
            "results": [r.to_dict() for r in self.results],
            "trunk_calibrated": self.proc.trunk0 is not None,
        }
