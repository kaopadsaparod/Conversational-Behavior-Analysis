"""Discourse Consistency Engine - Visual Kinematic Facial Feature Extraction.

Extracts normalized lip aperture, Eye Aspect Ratio (EAR), blink events, and head pose
quality metrics. Enforces lazy imports for mediapipe and opencv.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple, Union

import numpy as np

from dce.schemas import VisionFeatures

logger = logging.getLogger(__name__)


def compute_eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """Calculate Eye Aspect Ratio (EAR) from 6 2D facial landmark points.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye_landmarks: 6x2 array of eye landmark coordinates.

    Returns:
        EAR float value.
    """
    if len(eye_landmarks) < 6:
        return 0.0

    p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def estimate_head_pose_euler(
    coords: np.ndarray,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[float, float, float]]:
    """Estimate head pose Euler angles (pitch, yaw, roll) in degrees via cv2.solvePnP.

    Uses 6 canonical MediaPipe Face Mesh landmarks:
    nose tip (1), chin (152), left eye outer corner (33), right eye outer corner (263),
    left mouth corner (61), right mouth corner (291).
    """
    if len(coords) <= 291:
        return None

    try:
        import cv2
    except ImportError:
        return None

    image_points = np.array(
        [
            coords[1],    # Nose tip
            coords[152],  # Chin
            coords[33],   # Left eye outer corner
            coords[263],  # Right eye outer corner
            coords[61],   # Left mouth corner
            coords[291],  # Right mouth corner
        ],
        dtype=np.float64,
    )

    model_points = np.array(
        [
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye outer corner
            (225.0, 170.0, -135.0),   # Right eye outer corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ],
        dtype=np.float64,
    )

    focal_length = float(max(image_width, 1))
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    if not (math.isfinite(pitch) and math.isfinite(yaw) and math.isfinite(roll)):
        return None
    return pitch, yaw, roll


def extract_vision_frame(
    frame_rgb: np.ndarray,
    ear_blink_threshold: float = 0.20,
    max_head_pose_deg: float = 30.0,
) -> VisionFeatures:
    """Extract facial kinematics from a single video frame.

    Lazy imports mediapipe and cv2 to avoid heavy top-level imports.

    Args:
        frame_rgb: RGB image numpy array of shape (H, W, 3).
        ear_blink_threshold: EAR threshold below which an eye is deemed closed.

    Returns:
        VisionFeatures dataclass instance.
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return VisionFeatures(
            blink_rate_per_min=None,
            mean_lip_aperture=None,
            face_detected=False,
            quality_flag="empty_frame",
        )

    # LAZY IMPORT of mediapipe and cv2
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        logger.warning("mediapipe or opencv not installed; visual extraction unavailable.")
        return VisionFeatures(
            blink_rate_per_min=None,
            mean_lip_aperture=None,
            face_detected=False,
            quality_flag="no_vision_backend",
        )

    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(frame_rgb)
            if not results.multi_face_landmarks:
                return VisionFeatures(
                    blink_rate_per_min=None,
                    mean_lip_aperture=None,
                    face_detected=False,
                    quality_flag="no_face",
                )

            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame_rgb.shape
            coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)

            # Head-pose Euler angle gate (±30 deg)
            euler = estimate_head_pose_euler(coords, w, h)
            if euler is not None:
                pitch_deg, yaw_deg, _ = euler
                if abs(yaw_deg) > max_head_pose_deg or abs(pitch_deg) > max_head_pose_deg:
                    return VisionFeatures(
                        blink_rate_per_min=None,
                        mean_lip_aperture=None,
                        face_detected=True,
                        face_id_hint=None,
                        quality_flag="extreme_head_pose",
                    )

            # Inner lip aperture: distance between upper inner lip (13) and lower inner lip (14)
            # Normalized by horizontal lip width: distance between corners (61, 291)
            lip_top = coords[13]
            lip_bottom = coords[14]
            lip_left = coords[61]
            lip_right = coords[291]

            lip_height = float(np.linalg.norm(lip_top - lip_bottom))
            lip_width = float(np.linalg.norm(lip_left - lip_right))
            norm_lip_aperture = lip_height / max(lip_width, 1.0)

            # Left eye EAR: landmarks [33, 160, 158, 133, 153, 144]
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye_pts = coords[left_eye_indices]
            ear_left = compute_eye_aspect_ratio(left_eye_pts)

            # Right eye EAR: landmarks [362, 385, 387, 263, 373, 380]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_pts = coords[right_eye_indices]
            ear_right = compute_eye_aspect_ratio(right_eye_pts)

            mean_ear = 0.5 * (ear_left + ear_right)
            is_blink = mean_ear < ear_blink_threshold

            return VisionFeatures(
                blink_rate_per_min=None,
                mean_lip_aperture=norm_lip_aperture,
                face_detected=True,
                face_id_hint=None,
                quality_flag="ok",
            )

    except Exception as exc:
        logger.warning("Visual extraction error: %s", exc)
        return VisionFeatures(
            blink_rate_per_min=None,
            mean_lip_aperture=None,
            face_detected=False,
            quality_flag="extraction_error",
        )
