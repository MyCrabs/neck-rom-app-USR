"""Sign-convention tests for neck_rom.geometry (no camera needed).

The physical meaning of each sign is checked independently of the Euler
decomposition: e.g. "turn left" is asserted by the nose moving to the RIGHT of
the un-mirrored image (subject's left), not only by yaw > 0.
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neck_rom import geometry as g  # noqa: E402

W, H = 640, 480
CENTER = np.array([W / 2, -H / 2, 0.0])   # image centre in camera coords

# Frontal face in camera coordinates (pixels), head frame == camera frame:
# +X = subject's left, +Y = up, +Z = towards the camera.
FACE = {
    g.R_EYE_OUTER: (-50, 0, -6),
    g.R_EYE_INNER: (-18, 0, 0),
    g.L_EYE_INNER: (18, 0, 0),
    g.L_EYE_OUTER: (50, 0, -6),
    g.FOREHEAD: (0, 65, -6),   # same depth as the chin -> u exactly vertical
    g.CHIN: (0, -95, -6),
    g.NOSE_TIP: (0, -30, 30),
}


def make_landmarks(R=np.eye(3)):
    """Rotate the face, convert camera coords -> MediaPipe normalized landmarks."""
    lms = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(478)]
    for i, p in FACE.items():
        X, Y, Z = R @ np.array(p, float) + CENTER
        lms[i] = SimpleNamespace(x=X / W, y=-Y / H, z=-Z / W)   # inverse of _to_camera
    return lms


def head_angles(R_true):
    lms = make_landmarks(R_true)
    return g.angles_from_matrix(g.head_matrix_from_landmarks(lms, W, H)), lms


def test_roundtrip_random():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = rng.uniform(-60, 60, 3)
        b = g.angles_from_matrix(g.matrix_from_angles(*a))
        assert np.allclose(a, b, atol=1e-9)


def test_frontal_face_is_identity():
    lms = make_landmarks()
    R = g.head_matrix_from_landmarks(lms, W, H)
    assert np.allclose(R, np.eye(3), atol=1e-9)


def test_turn_left_gives_positive_yaw():
    # rotation about the vertical axis that swings the face towards +X (subject's left)
    (yaw, pitch, roll), lms = head_angles(g.rot_y(30))
    front = make_landmarks()
    assert lms[g.NOSE_TIP].x > front[g.NOSE_TIP].x   # nose moves to image right = subject's left
    assert yaw == pytest.approx(30, abs=1e-6)
    assert abs(pitch) < 1e-6 and abs(roll) < 1e-6


def test_look_up_gives_positive_pitch():
    (yaw, pitch, roll), lms = head_angles(g.rot_x(-20))
    front = make_landmarks()
    assert lms[g.NOSE_TIP].y < front[g.NOSE_TIP].y   # nose moves up in the image
    assert pitch == pytest.approx(20, abs=1e-6)
    assert abs(yaw) < 1e-6 and abs(roll) < 1e-6


def test_left_ear_to_left_shoulder_gives_positive_roll():
    (yaw, pitch, roll), lms = head_angles(g.rot_z(-25))
    front = make_landmarks()
    # top of the head goes towards the subject's left = image right
    assert lms[g.FOREHEAD].x > front[g.FOREHEAD].x
    # subject's left eye goes down (larger image y)
    assert lms[g.L_EYE_OUTER].y > front[g.L_EYE_OUTER].y
    assert roll == pytest.approx(25, abs=1e-6)
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6


def test_image_rotated_ccw_gives_negative_roll():
    """Same as the portrait.jpg check: rotate the image CCW by 20 deg -> roll ~ -20."""
    lms = make_landmarks()
    M = cv2.getRotationMatrix2D((W / 2, H / 2), 20, 1.0)   # positive angle = CCW on screen
    for lm in lms:
        x, y = lm.x * W, lm.y * H
        nx, ny = M @ np.array([x, y, 1.0])
        lm.x, lm.y = nx / W, ny / H
    _, _, roll = g.angles_from_matrix(g.head_matrix_from_landmarks(lms, W, H))
    assert roll == pytest.approx(-20, abs=1e-6)


@pytest.mark.parametrize("mov", [(40, 0, 0), (-40, 0, 0), (0, 35, 0), (0, -35, 0),
                                 (0, 0, 30), (0, 0, -30), (20, 10, -15)])
def test_relative_angles_with_offset_neutral(mov):
    R0 = g.rot_y(15) @ g.rot_x(-5)          # person sits 15 deg turned, slightly looking up
    R = R0 @ g.matrix_from_angles(*mov)     # movement in the person's own frame
    rel = g.relative_angles(R, R0)
    assert np.allclose(rel, mov, atol=1e-6)


def test_relative_angles_through_landmarks():
    R0 = g.rot_y(15)
    lm0 = make_landmarks(R0)
    lm1 = make_landmarks(R0 @ g.rot_y(40))
    Rm0 = g.head_matrix_from_landmarks(lm0, W, H)
    Rm1 = g.head_matrix_from_landmarks(lm1, W, H)
    yaw, pitch, roll = g.relative_angles(Rm1, Rm0)
    assert yaw == pytest.approx(40, abs=1e-6)


def test_average_rotation():
    mats = [g.matrix_from_angles(a, -a / 2, a / 3) for a in (-2, -1, 0, 1, 2)]
    Rm = g.average_rotation(mats)
    assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-9)
    assert np.allclose(g.angles_from_matrix(Rm), (0, 0, 0), atol=0.05)


def _pose(lx, ly, lz, rx, ry, rz, vis=0.99):
    pts = [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=vis) for _ in range(33)]
    pts[g.POSE_L_SHOULDER] = SimpleNamespace(x=lx, y=ly, z=lz, visibility=vis)
    pts[g.POSE_R_SHOULDER] = SimpleNamespace(x=rx, y=ry, z=rz, visibility=vis)
    return pts


def test_trunk_neutral():
    world = _pose(0.18, -0.4, 0.0, -0.18, -0.4, 0.0)
    norm = _pose(0.6, 0.6, 0, 0.4, 0.6, 0)
    yaw, roll = g.trunk_angles(world, norm, W, H)
    assert abs(yaw) < 1e-9 and abs(roll) < 1e-9


def test_trunk_left_shoulder_back_gives_positive_yaw():
    # subject's left shoulder (image right, larger x) moves AWAY from the camera (z grows)
    world = _pose(0.17, -0.4, 0.06, -0.17, -0.4, -0.06)
    norm = _pose(0.6, 0.6, 0, 0.4, 0.6, 0)
    yaw, _ = g.trunk_angles(world, norm, W, H)
    assert yaw > 0
    assert yaw == pytest.approx(math.degrees(math.atan2(0.12, 0.34)))


def test_trunk_left_shoulder_lower_gives_positive_roll():
    world = _pose(0.18, -0.4, 0.0, -0.18, -0.4, 0.0)
    norm = _pose(0.6, 0.65, 0, 0.4, 0.6, 0)    # left shoulder lower (y down)
    _, roll = g.trunk_angles(world, norm, W, H)
    assert roll > 0


def test_trunk_invisible_returns_none():
    world = _pose(0.18, -0.4, 0.0, -0.18, -0.4, 0.0, vis=0.2)
    norm = _pose(0.6, 0.6, 0, 0.4, 0.6, 0, vis=0.2)
    assert g.trunk_angles(world, norm, W, H) is None


def test_wrap_deg():
    assert g.wrap_deg(190) == pytest.approx(-170)
    assert g.wrap_deg(-190) == pytest.approx(170)
    assert g.wrap_deg(10) == pytest.approx(10)
