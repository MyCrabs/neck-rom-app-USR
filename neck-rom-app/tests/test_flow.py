"""Run the guided Flow with synthetic measurements at 30 fps (no camera needed)."""
import math
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
SHOULDER_W0 = 300.0                     # 2D shoulder width in pixels at rest
TOO_FAR_FACE_H = 0.12                   # tracker.py: face_h_frac (/ image height) < 0.12


def movement_angles(mv, amount):
    """(yaw, pitch, roll) for a movement of `amount` deg in its ROM direction."""
    a = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    a[mv.axis] = mv.sign * amount
    return a["yaw"], a["pitch"], a["roll"]


def synth(flow, t, trunk_extra_on_rot_left=20.0, no_trunk=False, pose_drift=0.0,
          face_height=0.35, drop_every=0):
    """Measurement for the current phase of the flow (smooth ramps, 1.5 s).

    trunk_extra_on_rot_left: real trunk turn during rot_left (head = neck + trunk);
        the 2D shoulder width narrows by cos(trunk) as it would on camera.
    pose_drift: fake trunk angle (fraction of the head angle) on the movement's
        own axis, WITHOUT any change of shoulder width -- the MediaPipe Pose
        artifact where shoulder depth follows the head.
    drop_every: every n-th frame during measure has no face (to count rejections).
    """
    mv = flow.movement
    el = flow.elapsed(t)
    amount, trunk_amount = 0.0, 0.0
    if flow.phase in ("measure", "return") and mv is not None:
        ramp = min(1.0, el / 1.5) if flow.phase == "measure" else max(0.0, 1 - el / 1.5)
        extra = trunk_extra_on_rot_left if mv.key == "rot_left" else 0.0
        amount = (TARGET + extra) * ramp          # head = neck + trunk
        trunk_amount = extra * ramp
        if drop_every and flow.phase == "measure" and int(round(el * FPS)) % drop_every == 0:
            return Measurement(face_found=False, quality_msg="q_no_face")
    y, p, r = movement_angles(mv, amount) if mv else (0.0, 0.0, 0.0)
    R = R_SEAT @ g.matrix_from_angles(y, p, r)
    trunk, width = None, None
    if not no_trunk:
        sign = mv.sign if mv else 1
        on_yaw = bool(mv and mv.axis == "yaw")
        on_roll = bool(mv and mv.axis == "roll")
        ty = TRUNK_SEAT[0] + sign * ((trunk_amount if on_yaw else 0.0)
                                     + (pose_drift * amount if on_yaw else 0.0))
        tr = TRUNK_SEAT[1] + sign * (pose_drift * amount if on_roll else 0.0)
        trunk = (ty, tr)
        width = SHOULDER_W0 * math.cos(math.radians(trunk_amount if on_yaw else 0.0))
    return Measurement(face_found=True, R=R, nose_px=(320, 240), face_h_frac=face_height,
                       trunk=trunk, shoulder_w=width,
                       quality_msg="q_too_far" if face_height < TOO_FAR_FACE_H else None)


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
        assert res[key].rom == res[key].rom_head, key   # no compensation -> rom = head
        assert res[key].rejected == {}, key
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


def test_small_pose_drift_is_ignored():
    """Fake trunk angle of 15% of the head angle (6 deg) stays inside the deadband."""
    flow = run_flow(trunk_extra_on_rot_left=0.0, pose_drift=0.15)
    for r in flow.results:
        assert r.compensation is False, r.key
        assert r.rom == r.rom_head == pytest.approx(TARGET, abs=1.5), r.key


def test_head_only_rotation_with_pose_drift_is_not_compensation():
    """Pose reports a 20 deg trunk turn but the 2D shoulder width does not change."""
    flow = run_flow(trunk_extra_on_rot_left=0.0, pose_drift=0.5)
    res = {r.key: r for r in flow.results}
    for key in ("rot_left", "rot_right"):
        r = res[key]
        assert r.max_trunk_raw == pytest.approx(20, abs=1.5), key
        assert r.max_trunk == 0.0, key
        assert r.compensation is False, key
        assert r.rom == r.rom_head == pytest.approx(TARGET, abs=1.5), key
    # tilts have no width check, but never lose more than half of the head angle
    for key in ("tilt_left", "tilt_right"):
        r = res[key]
        assert r.rom_neck >= 0.5 * r.rom_head - 0.5, key


def test_trunk_subtraction_is_capped_at_half_the_head_angle():
    flow = run_flow(trunk_extra_on_rot_left=60.0)       # head 100 = neck 40 + trunk 60
    rl = next(r for r in flow.results if r.key == "rot_left")
    assert rl.compensation is True
    assert rl.rom_head == pytest.approx(100, abs=1.5)
    assert rl.rom == rl.rom_neck == pytest.approx(50, abs=1.5)


def _measure_rot_left(head_yaw, trunk_yaw, shoulders_narrow):
    """Calibrate with shoulder_w=300 px, then hold rot_left at the given pose.

    shoulders_narrow: True -> shoulder_w = 300*cos(trunk yaw) (real trunk turn);
    False -> shoulder_w stays 300 (Pose trunk angle is an artifact of the head).
    Angles ramp in over 1 s (to stay below the outlier-jump check), then hold 3 s.
    """
    neutral = Measurement(face_found=True, R=R_SEAT, nose_px=(320, 240), face_h_frac=0.35,
                          trunk=TRUNK_SEAT, shoulder_w=300.0)
    flow = S.Flow(S.AngleProcessor())
    t = 0.0
    flow.start(t)
    while flow.phase in ("calib_prep", "calib"):
        flow.update(neutral, t)
        t += 1 / FPS
    assert flow.proc.width0 == pytest.approx(300.0)
    while flow.movement.key != "rot_left":
        flow.skip(t)
    flow.next(t)                                   # prep -> measure
    flow.update(neutral, t)
    assert flow.phase == "measure" and flow.movement.key == "rot_left"
    t0 = t
    while t - t0 < 4.0:
        k = min(1.0, (t - t0) / 1.0)
        sw = 300.0 * math.cos(math.radians(trunk_yaw * k)) if shoulders_narrow else 300.0
        m = Measurement(face_found=True, R=R_SEAT @ g.matrix_from_angles(head_yaw * k, 0, 0),
                        nose_px=(320, 240), face_h_frac=0.35,
                        trunk=(TRUNK_SEAT[0] + trunk_yaw * k, TRUNK_SEAT[1]), shoulder_w=sw)
        t += 1 / FPS
        flow.update(m, t)
    flow.next(t)                                   # end the measurement
    return flow.results[-1]


def test_real_trunk_rotation_narrows_shoulders_and_is_compensation():
    # head 60 = neck 40 + trunk 20; shoulders narrow to 300*cos(20deg) px
    r = _measure_rot_left(60.0, 20.0, shoulders_narrow=True)
    assert r.compensation is True
    assert r.max_trunk == pytest.approx(20, abs=1.5)
    assert r.rom == r.rom_neck == pytest.approx(40, abs=1.5)


def test_head_only_rotation_keeps_shoulder_width_and_is_not_compensation():
    # Pose reports trunk yaw 20 deg, but the 2D shoulder width stays at 300 px
    r = _measure_rot_left(40.0, 20.0, shoulders_narrow=False)
    assert r.max_trunk_raw == pytest.approx(20, abs=1.5)
    assert r.max_trunk == 0.0
    assert r.compensation is False
    assert r.rom == r.rom_head == pytest.approx(40, abs=1.5)


def test_too_far_is_a_hint_not_a_rejection():
    flow = run_flow(face_height=0.08)
    for r in flow.results:
        assert r.valid_frames > 150 and r.rejected == {}, r.key
        assert r.rom == pytest.approx(TARGET, abs=1.5), r.key
    s = S.AngleProcessor().update(synth(flow, 0.0, face_height=0.08), 0.0)
    assert s.valid and s.hint == "q_too_far"


def test_rejected_frames_are_counted_per_reason():
    flow = run_flow(drop_every=10)
    for r in flow.results:
        n = r.rejected.get("q_no_face", 0)
        assert n >= 15, r.key
        assert r.valid_frames + sum(r.rejected.values()) == r.total_frames, r.key
    assert "rejected" in flow.to_dict()["results"][0]


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
