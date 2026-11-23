"""MediaPipe Face Landmarker + Pose Landmarker wrapper.

Produces one `Measurement` per frame with the raw head rotation matrix, the
shoulder-line (trunk) angles and simple quality flags.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from . import geometry as g

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"


@dataclass
class Measurement:
    face_found: bool = False
    R: np.ndarray | None = None          # head rotation (camera frame)
    nose_px: tuple[int, int] | None = None
    eye_dist_frac: float = 0.0           # inter-ocular distance / image width
    face_h_frac: float = 0.0             # forehead-chin distance / image height
    shoulder_w: float | None = None      # 2D shoulder width in pixels
    trunk: tuple[float, float] | None = None   # (trunk_yaw, trunk_roll) absolute
    shoulders_px: tuple[tuple[int, int], tuple[int, int]] | None = None  # (L, R)
    quality_msg: str | None = None       # key into i18n, e.g. "q_no_face"


class Tracker:
    def __init__(self, use_pose: bool = True, face_model: Path = FACE_MODEL,
                 pose_model: Path = POSE_MODEL, face_conf: float = 0.3,
                 pose_every: int = 2, infer_width: int = 640):
        """face_conf: lower = keeps tracking the face at large angles longer.
        pose_every: run Pose only every N frames (shoulders move slowly) to
        raise FPS; a higher FPS also helps the face tracker not lose the face.
        infer_width: frames are downscaled to this width before inference
        (landmarks are normalized, so results are unchanged)."""
        self.pose_every, self.infer_width = max(1, pose_every), infer_width
        self._frame_i = 0
        self._last_pose = None
        for p in [face_model] + ([pose_model] if use_pose else []):
            if not Path(p).exists():
                raise FileNotFoundError(
                    f"Missing model file: {p}\nRun:  python download_models.py")
        base = mp.tasks.BaseOptions
        self.face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(face_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=face_conf,
                min_face_presence_confidence=face_conf,
                min_tracking_confidence=face_conf,
            ))
        self.pose = None
        if use_pose:
            self.pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base(model_asset_path=str(pose_model)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                ))
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Measurement:
        # MediaPipe VIDEO mode needs strictly increasing timestamps
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        h, w = frame_bgr.shape[:2]
        small = frame_bgr
        if self.infer_width and w > self.infer_width:
            small = cv2.resize(frame_bgr, (self.infer_width, round(h * self.infer_width / w)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        m = Measurement()

        fr = self.face.detect_for_video(image, ts)
        if fr.face_landmarks:
            lms = fr.face_landmarks[0]
            m.face_found = True
            m.R = g.head_matrix_from_landmarks(lms, w, h)
            nose = lms[g.NOSE_TIP]
            m.nose_px = (int(nose.x * w), int(nose.y * h))
            le, re = lms[g.L_EYE_OUTER], lms[g.R_EYE_OUTER]
            m.eye_dist_frac = math.hypot(le.x - re.x, (le.y - re.y) * h / w)
            fh, ch = lms[g.FOREHEAD], lms[g.CHIN]
            m.face_h_frac = math.hypot((fh.x - ch.x) * w / h, fh.y - ch.y)
            # Only a hint for the user. Eye distance shrinks when the head
            # turns, so it must NOT be used to reject frames (that cut off
            # the maximum angle). Face height is ~yaw-invariant.
            if m.face_h_frac < 0.12:
                m.quality_msg = "q_too_far"
        else:
            m.quality_msg = "q_no_face"

        if self.pose is not None:
            if self._frame_i % self.pose_every == 0 or self._last_pose is None:
                self._last_pose = self.pose.detect_for_video(image, ts)
            self._frame_i += 1
            pr = self._last_pose
            if pr.pose_landmarks and pr.pose_world_landmarks:
                pn, pw = pr.pose_landmarks[0], pr.pose_world_landmarks[0]
                m.trunk = g.trunk_angles(pw, pn, w, h)
                ls, rs = pn[g.POSE_L_SHOULDER], pn[g.POSE_R_SHOULDER]
                m.shoulders_px = ((int(ls.x * w), int(ls.y * h)),
                                  (int(rs.x * w), int(rs.y * h)))
                if m.trunk is not None:
                    m.shoulder_w = math.hypot((ls.x - rs.x) * w, (ls.y - rs.y) * h)
        return m

    def close(self):
        self.face.close()
        if self.pose is not None:
            self.pose.close()
