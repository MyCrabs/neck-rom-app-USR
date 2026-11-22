"""Run the guided Flow with synthetic measurements at 30 fps (no camera needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neck_rom import geometry as g  # noqa: E402
from neck_rom import session as S  # noqa: E402
from neck_rom.tracker import Measurement  # noqa: E402

FPS = 30.0
TARGET = 40.0
R_SEAT = g.rot_y(6) @ g.rot_x(-4)      # person not exactly facing the camera
TRUNK_SEAT = (3.0, -2.0)                # absolute shoulder angles at rest


def movement_angles(mv, amount):
    """(yaw, pitch, roll) for a movement of `amount` deg in its ROM direction."""
    a = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    a[mv.axis] = mv.sign * amount
    return a["yaw"], a["pitch"], a["roll"]


def synth(flow, t, trunk_extra_on_rot_left=20.0, no_trunk=False):
    """Measurement for the current phase of the flow (smooth ramps, 1.5 s)."""
    mv = flow.movement
    el = flow.elapsed(t)
    amount, trunk_amount = 0.0, 0.0
    if flow.phase in ("measure", "return") and mv is not None:
        ramp = min(1.0, el / 1.5) if flow.phase == "measure" else max(0.0, 1 - el / 1.5)
        extra = trunk_extra_on_rot_left if mv.key == "rot_left" else 0.0
        amount = (TARGET + extra) * ramp          # head = neck + trunk
        trunk_amount = extra * ramp
    y, p, r = movement_angles(mv, amount) if mv else (0.0, 0.0, 0.0)
    R = R_SEAT @ g.matrix_from_angles(y, p, r)
    trunk = None
    if not no_trunk:
        ty = TRUNK_SEAT[0] + (trunk_amount if mv and mv.axis == "yaw" else 0.0) * (mv.sign if mv else 1)
        trunk = (ty, TRUNK_SEAT[1])
    return Measurement(face_found=True, R=R, nose_px=(320, 240), eye_dist_frac=0.15, trunk=trunk)


def run_flow(**kw):
    flow = S.Flow(S.AngleProcessor())
    t, started = 0.0, False
    while t < 300 and not flow.finished:
        if not started and t >= 0.5:
            flow.start(t)
            started = True
        flow.update(synth(flow, t, **kw), t)
        t += 1 / FPS
    return flow


def test_full_flow_with_trunk_compensation():
    flow = run_flow()
    assert flow.finished
    assert flow.proc.trunk0 is not None
    res = {r.key: r for r in flow.results}
    assert list(res) == [m.key for m in S.MOVEMENTS]
    for key, r in res.items():
        assert not r.skipped
        assert r.rom == pytest.approx(TARGET, abs=1.5), key
        assert r.valid_frames > 150
    rl = res["rot_left"]
    assert rl.compensation is True
    assert rl.rom_head == pytest.approx(60, abs=1.5)
    assert rl.rom_neck == pytest.approx(40, abs=1.5)
    assert rl.max_trunk == pytest.approx(20, abs=1.5)
    for key in ("flexion", "extension", "tilt_left", "tilt_right", "rot_right"):
        assert res[key].compensation is False, key
    # flexion / extension have no trunk compensation axis
    assert res["flexion"].rom_neck is None and res["extension"].rom_neck is None
    d = flow.to_dict()
    assert len(d["results"]) == 6 and d["trunk_calibrated"] is True


def test_flow_without_shoulders_reports_head_rom():
    flow = run_flow(no_trunk=True)
    assert flow.finished and flow.proc.trunk0 is None
    rl = next(r for r in flow.results if r.key == "rot_left")
    assert rl.rom_neck is None
    assert rl.rom == rl.rom_head == pytest.approx(60, abs=1.5)


def test_skip_and_next():
    flow = S.Flow(S.AngleProcessor())
    t = 0.0
    flow.start(t)
    # calibrate
    while flow.phase in ("calib_prep", "calib"):
        flow.update(synth(flow, t), t)
        t += 1 / FPS
    assert flow.phase == "prep"
    flow.skip(t)                       # skip flexion
    assert flow.results[-1].skipped and flow.idx == 1
    flow.update(synth(flow, t), t)
    flow.next(t)                       # SPACE in prep -> timer expires
    t += 1 / FPS
    flow.update(synth(flow, t), t)
    assert flow.phase == "measure"


def test_calibration_retries_when_face_missing():
    flow = S.Flow(S.AngleProcessor())
    t = 0.0
    flow.start(t)
    for _ in range(int(6 * FPS)):
        flow.update(Measurement(face_found=False, quality_msg="q_no_face"), t)
        t += 1 / FPS
    assert flow.phase in ("calib_prep", "calib")
    assert flow.message == "msg_calib_noface"
    assert not flow.proc.calibrated
