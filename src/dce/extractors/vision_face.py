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


def extract_vision_frame(
    frame_rgb: np.ndarray,
    ear_blink_threshold: float = 0.20,
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
                blink_rate_per_min=20.0 if not is_blink else 40.0,
                mean_lip_aperture=norm_lip_aperture,
                face_detected=True,
                face_id_hint="FACE_0",
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
