"""Head / trunk angle math.

Coordinate convention used everywhere in this project (camera frame):
    X -> right side of the (un-mirrored) camera image
    Y -> up
    Z -> towards the camera (out of the screen)

When the user faces the camera, the subject's LEFT side appears on the RIGHT of
the un-mirrored camera image, so +X == subject's left.

Head frame (columns of R, expressed in camera coordinates):
    r = subject's right eye -> subject's left eye   (points to subject's left)
    u = chin -> forehead                             (points up)
    f = r x u                                        (points out of the face)

Euler angles (degrees), all signs chosen so that they match the ROM formulas
in the project spec (section 4.3):
    yaw   > 0 : head turned to the subject's LEFT     (rotation left)
    pitch > 0 : looking UP                            (extension)
    roll  > 0 : left ear towards left shoulder        (tilt / lateral flexion left)

Decomposition: R = Ry(yaw) * Rx(-pitch) * Rz(-roll)   (standard right-handed
rotation matrices).  See tests/test_geometry.py for checks.
"""
from __future__ import annotations

import math

import numpy as np

# MediaPipe Face Mesh landmark indices (subject's left/right)
R_EYE_OUTER, R_EYE_INNER = 33, 133
L_EYE_INNER, L_EYE_OUTER = 362, 263
FOREHEAD, CHIN, NOSE_TIP = 10, 152, 1

# MediaPipe Pose landmark indices (subject's left/right)
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def matrix_from_angles(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of angles_from_matrix (used by tests and for drawing)."""
    return rot_y(yaw) @ rot_x(-pitch) @ rot_z(-roll)


def angles_from_matrix(R: np.ndarray) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees with the project's sign convention."""
    yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
    pitch = math.degrees(math.asin(float(np.clip(R[1, 2], -1.0, 1.0))))
    roll = -math.degrees(math.atan2(R[1, 0], R[1, 1]))
    return yaw, pitch, roll


def _to_camera(lm, w: int, h: int) -> np.ndarray:
    """MediaPipe normalized landmark -> camera-frame point in pixel units.

    MediaPipe: x right, y DOWN, z smaller = closer to camera, z scaled like x.
    """
    return np.array([lm.x * w, -lm.y * h, -lm.z * w], dtype=float)


def head_matrix_from_landmarks(landmarks, w: int, h: int) -> np.ndarray:
    """Build the head rotation matrix R (columns r, u, f) from 478 face landmarks."""
    p = lambda i: _to_camera(landmarks[i], w, h)  # noqa: E731
    right_eye = (p(R_EYE_OUTER) + p(R_EYE_INNER)) / 2.0
    left_eye = (p(L_EYE_OUTER) + p(L_EYE_INNER)) / 2.0
    r = left_eye - right_eye
    u = p(FOREHEAD) - p(CHIN)
    r /= np.linalg.norm(r)
    u = u - np.dot(u, r) * r  # Gram-Schmidt
    u /= np.linalg.norm(u)
    f = np.cross(r, u)
    return np.column_stack([r, u, f])


def relative_angles(R: np.ndarray, R0: np.ndarray) -> tuple[float, float, float]:
    """Head angles relative to the calibrated neutral pose R0.

    R_rel = R0^T R is the rotation expressed in the neutral head frame, i.e. the
    person's own anatomical axes. This is the matrix version of theta - theta0
    and stays correct even if the person does not sit exactly facing the camera.
    """
    return angles_from_matrix(R0.T @ R)


def average_rotation(mats: list[np.ndarray]) -> np.ndarray:
    """Average of rotation matrices (projected back to SO(3) with SVD)."""
    M = np.mean(np.stack(mats), axis=0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def trunk_angles(pose_world, pose_norm, w: int, h: int):
    """Shoulder-line angles (degrees) or None if shoulders are not visible.

    trunk_yaw  > 0 : shoulders/trunk rotated to the subject's LEFT
    trunk_roll > 0 : trunk leaning to the subject's LEFT (left shoulder lower)
    """
    ls_n, rs_n = pose_norm[POSE_L_SHOULDER], pose_norm[POSE_R_SHOULDER]
    vis = min(getattr(ls_n, "visibility", 1.0) or 0.0, getattr(rs_n, "visibility", 1.0) or 0.0)
    if vis < 0.5:
        return None
    ls_w, rs_w = pose_world[POSE_L_SHOULDER], pose_world[POSE_R_SHOULDER]
    # world landmarks: metres, x right, y down, z away from camera.
    # Turning left moves the left shoulder away from the camera (z_L grows).
    yaw = math.degrees(math.atan2(ls_w.z - rs_w.z, ls_w.x - rs_w.x))
    # image coordinates for roll (2D shoulder line is reliable)
    roll = math.degrees(math.atan2((ls_n.y - rs_n.y) * h, (ls_n.x - rs_n.x) * w))
    return yaw, roll


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0
