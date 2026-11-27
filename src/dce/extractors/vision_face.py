"""Discourse Consistency Engine - Visual Kinematic Facial Feature Extraction.

Extracts normalized lip aperture, Eye Aspect Ratio (EAR), blink events, and head pose
quality metrics. Enforces lazy imports for mediapipe and opencv.
"""

from __future__ import annotations

import logging
import math
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from dce.schemas import VisionFeatures

logger = logging.getLogger(__name__)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_FACE_LANDMARKER_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "face_landmarker.task"
)

_LANDMARKER_LOCK = threading.Lock()
_IMAGE_LANDMARKER_CACHE: Dict[Tuple[Any, str], Any] = {}


def clear_face_landmarker_cache() -> None:
    """Close and clear cached stateless IMAGE-mode FaceLandmarker instances."""
    with _LANDMARKER_LOCK:
        for landmarker in _IMAGE_LANDMARKER_CACHE.values():
            close_fn = getattr(landmarker, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        _IMAGE_LANDMARKER_CACHE.clear()


def ensure_face_landmarker_model(model_path: Optional[Union[str, Path]] = None) -> Path:
    """Return local path to face_landmarker.task, downloading it once if missing."""
    target = Path(model_path) if model_path is not None else DEFAULT_FACE_LANDMARKER_PATH
    with _LANDMARKER_LOCK:
        if target.is_file() and target.stat().st_size > 0:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(".task.tmp")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, str(tmp_path))
        tmp_path.replace(target)
        return target


def _get_image_face_landmarker(mp_module: Any, model_path: Path) -> Any:
    """Return a cached stateless RunningMode.IMAGE FaceLandmarker singleton."""
    cache_key = (mp_module, str(model_path))
    with _LANDMARKER_LOCK:
        landmarker = _IMAGE_LANDMARKER_CACHE.get(cache_key)
        if landmarker is None:
            vision = mp_module.tasks.vision
            base_options = mp_module.tasks.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            landmarker = vision.FaceLandmarker.create_from_options(options)
            _IMAGE_LANDMARKER_CACHE[cache_key] = landmarker
        return landmarker


def _euler_from_transformation_matrix(matrix_4x4: Any) -> Optional[Tuple[float, float, float]]:
    """Extract folded (pitch, yaw, roll) in degrees from MediaPipe 4x4 facial transformation matrix."""
    try:
        import cv2

        mat = np.asarray(matrix_4x4, dtype=np.float64)
        if mat.shape != (4, 4):
            return None
        rmat = mat[:3, :3]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch = fold_euler_angle_deg(float(angles[0]))
        yaw = fold_euler_angle_deg(float(angles[1]))
        roll = fold_euler_angle_deg(float(angles[2]))
        if not (math.isfinite(pitch) and math.isfinite(yaw) and math.isfinite(roll)):
            return None
        return pitch, yaw, roll
    except Exception:
        return None


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

    # Camera-aligned 3D face model (X right, Y down, Z into face) so frontal pose -> (0, 0, 0) deg
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, 330.0, 65.0),        # Chin
            (-225.0, -170.0, 135.0),   # Left eye outer corner
            (225.0, -170.0, 135.0),    # Right eye outer corner
            (-150.0, 150.0, 125.0),    # Left mouth corner
            (150.0, 150.0, 125.0),     # Right mouth corner
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
    pitch, yaw, roll = (
        fold_euler_angle_deg(float(angles[0])),
        fold_euler_angle_deg(float(angles[1])),
        fold_euler_angle_deg(float(angles[2])),
    )
    if not (math.isfinite(pitch) and math.isfinite(yaw) and math.isfinite(roll)):
        return None
    return pitch, yaw, roll


def fold_euler_angle_deg(angle_deg: float) -> float:
    """Fold OpenCV RQDecomp3x3 Euler angles (e.g. +-178 deg frontal flip) into [-90, +90] deg."""
    if not math.isfinite(angle_deg):
        return angle_deg
    wrapped = ((angle_deg + 180.0) % 360.0) - 180.0
    if wrapped > 90.0:
        wrapped -= 180.0
    elif wrapped < -90.0:
        wrapped += 180.0
    return wrapped


def evaluate_head_pose_quality(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float = 0.0,
    max_yaw_deg: float = 30.0,
    max_pitch_deg: float = 25.0,
    max_roll_deg: float = 30.0,
) -> Tuple[bool, str]:
    """Evaluate head pose angles against OpenFace-style quality gate thresholds.

    Rejects frames where speaker turns head > 30 deg, tilts > 30 deg, or pitches
    head down/up > 25 deg (e.g., looking down to read documents) to prevent
    spurious Blink/Lip Z-Score spikes.
    """
    if not (math.isfinite(pitch_deg) and math.isfinite(yaw_deg) and math.isfinite(roll_deg)):
        return False, "extreme_head_pose"
    pitch = fold_euler_angle_deg(pitch_deg)
    yaw = fold_euler_angle_deg(yaw_deg)
    roll = fold_euler_angle_deg(roll_deg)
    if abs(yaw) > max_yaw_deg or abs(pitch) > max_pitch_deg or abs(roll) > max_roll_deg:
        return False, "extreme_head_pose"
    return True, "ok"


def compute_eye_gaze_offset(coords: np.ndarray) -> Optional[Tuple[float, float]]:
    """Compute normalized horizontal and vertical iris gaze offset in [-1, 1] from MediaPipe iris landmarks."""
    if len(coords) < 478:
        return None

    def _single_eye_gaze(iris_idx: int, c_left: int, c_right: int, c_top: int, c_bot: int) -> Tuple[float, float]:
        iris = coords[iris_idx]
        left, right = coords[c_left], coords[c_right]
        top, bot = coords[c_top], coords[c_bot]
        w = max(float(np.linalg.norm(right - left)), 1.0)
        h = max(float(np.linalg.norm(bot - top)), 1.0)
        cx = 0.5 * (left[0] + right[0])
        cy = 0.5 * (top[1] + bot[1])
        return float((iris[0] - cx) / (0.5 * w)), float((iris[1] - cy) / (0.5 * h))

    lgx, lgy = _single_eye_gaze(468, 33, 133, 159, 145)
    rgx, rgy = _single_eye_gaze(473, 362, 263, 386, 374)
    gx = 0.5 * (lgx + rgx)
    gy = 0.5 * (lgy + rgy)
    return (
        max(-1.0, min(1.0, gx)),
        max(-1.0, min(1.0, gy)),
    )


def compute_facial_action_units(
    coords: np.ndarray,
    ear_blink_threshold: float = 0.20,
) -> Dict[str, float]:
    """Compute geometric Py-FEAT/OpenFace-inspired Action Units normalized by inter-ocular distance."""
    if len(coords) <= 291:
        return {}

    left_eye_outer = coords[33]
    right_eye_outer = coords[263]
    inter_ocular = max(float(np.linalg.norm(right_eye_outer - left_eye_outer)), 1.0)

    # AU01: Inner brow raise (distance from inner brow 107/336 to eye inner corner 133/362)
    au01 = float(
        0.5 * (np.linalg.norm(coords[107] - coords[133]) + np.linalg.norm(coords[336] - coords[362]))
        / inter_ocular
    )
    # AU04: Brow lowerer (inverse proxy: distance between inner brows 107 and 336)
    au04 = float(1.0 - min(1.0, np.linalg.norm(coords[107] - coords[336]) / inter_ocular))
    # AU12: Lip corner puller (mouth width 61-291 / inter_ocular)
    au12 = float(np.linalg.norm(coords[61] - coords[291]) / inter_ocular)
    # AU25: Lips part (inner lip height 13-14 / inter_ocular)
    au25 = float(np.linalg.norm(coords[13] - coords[14]) / inter_ocular)

    ear_l = compute_eye_aspect_ratio(coords[[33, 160, 158, 133, 153, 144]])
    ear_r = compute_eye_aspect_ratio(coords[[362, 385, 387, 263, 373, 380]])
    mean_ear = 0.5 * (ear_l + ear_r)
    au45 = 1.0 if mean_ear < ear_blink_threshold else 0.0

    return {
        "AU01_inner_brow_raise": round(au01, 4),
        "AU04_brow_lowerer": round(au04, 4),
        "AU12_lip_corner_puller": round(au12, 4),
        "AU25_lips_part": round(au25, 4),
        "AU45_blink": au45,
    }


def extract_vision_frame(
    frame_rgb: np.ndarray,
    ear_blink_threshold: float = 0.20,
    max_head_pose_deg: float = 30.0,
    max_pitch_deg: float = 25.0,
    min_luma: float = 20.0,
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
        # Low-light quality gate
        mean_luma = float(np.mean(frame_rgb))
        if mean_luma < min_luma:
            return VisionFeatures(
                blink_rate_per_min=None,
                mean_lip_aperture=None,
                face_detected=False,
                low_light_flag=True,
                quality_flag="low_light",
            )

        if frame_rgb.ndim == 2:
            frame_rgb = np.stack([frame_rgb, frame_rgb, frame_rgb], axis=-1)
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        h, w = frame_rgb.shape[:2]
        landmarks = None
        blendshapes_map: Dict[str, float] = {}
        transform_matrix: Any = None

        if hasattr(mp, "tasks"):
            model_path = ensure_face_landmarker_model()
            landmarker = _get_image_face_landmarker(mp, model_path)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
            results = landmarker.detect(mp_image)
            if not getattr(results, "face_landmarks", None):
                return VisionFeatures(
                    blink_rate_per_min=None,
                    mean_lip_aperture=None,
                    face_detected=False,
                    quality_flag="no_face",
                )
            landmarks = results.face_landmarks[0]
            if getattr(results, "face_blendshapes", None):
                blendshapes_map = {
                    str(b.category_name): float(b.score)
                    for b in results.face_blendshapes[0]
                    if getattr(b, "category_name", None)
                }
            if getattr(results, "facial_transformation_matrixes", None):
                transform_matrix = results.facial_transformation_matrixes[0]
        else:
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

        coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)
        return _build_vision_features_from_landmarks(
            coords=coords,
            image_width=w,
            image_height=h,
            ear_blink_threshold=ear_blink_threshold,
            max_head_pose_deg=max_head_pose_deg,
            max_pitch_deg=max_pitch_deg,
            blendshapes_map=blendshapes_map,
            transform_matrix=transform_matrix,
        )

    except Exception as exc:
        logger.warning("Visual extraction error: %s", exc)
        return VisionFeatures(
            blink_rate_per_min=None,
            mean_lip_aperture=None,
            face_detected=False,
            quality_flag="extraction_error",
        )


def _build_vision_features_from_landmarks(
    coords: np.ndarray,
    image_width: int,
    image_height: int,
    ear_blink_threshold: float = 0.20,
    max_head_pose_deg: float = 30.0,
    max_pitch_deg: float = 25.0,
    blendshapes_map: Optional[Dict[str, float]] = None,
    transform_matrix: Any = None,
) -> VisionFeatures:
    """Build VisionFeatures from 2D pixel landmarks + optional blendshapes/transformation matrix."""
    euler = (
        _euler_from_transformation_matrix(transform_matrix)
        if transform_matrix is not None
        else None
    )
    if euler is None:
        euler = estimate_head_pose_euler(coords, image_width, image_height)

    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    if euler is not None:
        pitch_deg, yaw_deg, roll_deg = euler
        pose_ok, pose_flag = evaluate_head_pose_quality(
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            max_yaw_deg=max_head_pose_deg,
            max_pitch_deg=max_pitch_deg,
            max_roll_deg=max_head_pose_deg,
        )
        if not pose_ok:
            return VisionFeatures(
                blink_rate_per_min=None,
                mean_lip_aperture=None,
                face_detected=True,
                face_id_hint=None,
                head_pose_yaw_deg=yaw_deg,
                head_pose_pitch_deg=pitch_deg,
                head_pose_roll_deg=roll_deg,
                quality_flag=pose_flag,
            )

    lip_top = coords[13]
    lip_bottom = coords[14]
    lip_left = coords[61]
    lip_right = coords[291]
    lip_height = float(np.linalg.norm(lip_top - lip_bottom))
    lip_width = float(np.linalg.norm(lip_left - lip_right))
    norm_lip_aperture = lip_height / max(lip_width, 1.0)

    gaze = compute_eye_gaze_offset(coords)
    action_units = compute_facial_action_units(coords, ear_blink_threshold=ear_blink_threshold)
    if blendshapes_map:
        blink_l = blendshapes_map.get("eyeBlinkLeft")
        blink_r = blendshapes_map.get("eyeBlinkRight")
        if blink_l is not None and blink_r is not None:
            action_units["AU45_blink_blendshape"] = round(0.5 * (blink_l + blink_r), 4)
            action_units["AU45_blink"] = 1.0 if 0.5 * (blink_l + blink_r) >= 0.45 else 0.0

    return VisionFeatures(
        blink_rate_per_min=None,
        mean_lip_aperture=norm_lip_aperture,
        face_detected=True,
        face_id_hint=None,
        head_pose_yaw_deg=yaw_deg,
        head_pose_pitch_deg=pitch_deg,
        head_pose_roll_deg=roll_deg,
        eye_gaze_horizontal=gaze[0] if gaze is not None else None,
        eye_gaze_vertical=gaze[1] if gaze is not None else None,
        action_units=action_units,
        low_light_flag=False,
        quality_flag="ok",
    )


def extract_vision_sequence(
    frames_rgb: Sequence[np.ndarray],
    fps: float = 15.0,
    ear_blink_threshold: float = 0.20,
    max_head_pose_deg: float = 30.0,
    max_pitch_deg: float = 25.0,
    min_luma: float = 20.0,
    max_faces: int = 1,
) -> Union[List[VisionFeatures], Dict[str, List[VisionFeatures]]]:
    """Process a contiguous video frame sequence using a fresh RunningMode.VIDEO FaceLandmarker.

    Creates a dedicated VIDEO-mode instance per sequence so timestamp_ms is strictly
    monotonic and isolated from other files, and computes blink_rate_per_min across
    valid frames.

    When ``max_faces == 1`` (default), returns ``List[VisionFeatures]``.
    When ``max_faces > 1``, tracks up to ``max_faces`` faces grouped by horizontal
    spatial center and returns ``Dict[str, List[VisionFeatures]]`` keyed by face track ID
    (e.g., ``"face_0"``, ``"face_1"``).
    """
    eff_max_faces = max(1, int(max_faces))
    if not frames_rgb:
        return [] if eff_max_faces == 1 else {}

    try:
        import cv2  # noqa: F401
        import mediapipe as mp
    except ImportError:
        no_backend = [
            VisionFeatures(face_detected=False, quality_flag="no_vision_backend")
            for _ in frames_rgb
        ]
        return no_backend if eff_max_faces == 1 else {"face_0": no_backend}

    if not hasattr(mp, "tasks"):
        seq_1 = [
            extract_vision_frame(
                f,
                ear_blink_threshold=ear_blink_threshold,
                max_head_pose_deg=max_head_pose_deg,
                max_pitch_deg=max_pitch_deg,
                min_luma=min_luma,
            )
            for f in frames_rgb
        ]
        if eff_max_faces == 1:
            return seq_1
        return {
            "face_0": [v.model_copy(update={"face_id_hint": "face_0"}) for v in seq_1]
        }

    model_path = ensure_face_landmarker_model()
    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=eff_max_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )

    eff_fps = max(fps, 1.0)
    step_ms = max(1, int(round(1000.0 / eff_fps)))
    win_frames = max(1, int(round(10.0 * eff_fps)))

    if eff_max_faces == 1:
        out_features: List[VisionFeatures] = []
        valid_flags: List[bool] = []
        blink_onsets: List[int] = []
        prev_blinking = False

        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            for idx, raw_frame in enumerate(frames_rgb):
                if raw_frame is None or raw_frame.size == 0:
                    out_features.append(VisionFeatures(face_detected=False, quality_flag="empty_frame"))
                    valid_flags.append(False)
                    blink_onsets.append(0)
                    continue
                if float(np.mean(raw_frame)) < min_luma:
                    out_features.append(
                        VisionFeatures(face_detected=False, low_light_flag=True, quality_flag="low_light")
                    )
                    valid_flags.append(False)
                    blink_onsets.append(0)
                    continue

                frame = np.stack([raw_frame, raw_frame, raw_frame], axis=-1) if raw_frame.ndim == 2 else raw_frame
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                h, w = frame.shape[:2]
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
                results = landmarker.detect_for_video(mp_image, idx * step_ms)
                if not getattr(results, "face_landmarks", None):
                    out_features.append(VisionFeatures(face_detected=False, quality_flag="no_face"))
                    valid_flags.append(False)
                    blink_onsets.append(0)
                    continue

                coords = np.array([[lm.x * w, lm.y * h] for lm in results.face_landmarks[0]], dtype=np.float64)
                blendshapes_map = (
                    {
                        str(b.category_name): float(b.score)
                        for b in results.face_blendshapes[0]
                        if getattr(b, "category_name", None)
                    }
                    if getattr(results, "face_blendshapes", None)
                    else {}
                )
                transform_matrix = (
                    results.facial_transformation_matrixes[0]
                    if getattr(results, "facial_transformation_matrixes", None)
                    else None
                )
                vf = _build_vision_features_from_landmarks(
                    coords=coords,
                    image_width=w,
                    image_height=h,
                    ear_blink_threshold=ear_blink_threshold,
                    max_head_pose_deg=max_head_pose_deg,
                    max_pitch_deg=max_pitch_deg,
                    blendshapes_map=blendshapes_map,
                    transform_matrix=transform_matrix,
                )
                is_valid = vf.quality_flag == "ok"
                onset = 0
                if is_valid:
                    is_blinking = vf.action_units.get("AU45_blink", 0.0) >= 0.5
                    if is_blinking and not prev_blinking:
                        onset = 1
                    prev_blinking = is_blinking
                valid_flags.append(is_valid)
                blink_onsets.append(onset)
                if is_valid:
                    lo = max(0, len(valid_flags) - win_frames)
                    win_valid = sum(valid_flags[lo:])
                    win_valid_sec = float(win_valid) / eff_fps
                    if win_valid_sec >= 6.0:
                        win_blinks = sum(blink_onsets[lo:])
                        vf = vf.model_copy(update={"blink_rate_per_min": float(win_blinks) / (win_valid_sec / 60.0)})
                out_features.append(vf)

        return out_features

    # Multi-face tracking grouped by horizontal spatial center (up to eff_max_faces)
    n_frames = len(frames_rgb)
    track_centers: Dict[str, float] = {}
    track_features: Dict[str, List[VisionFeatures]] = {}
    track_valid_flags: Dict[str, List[bool]] = {}
    track_blink_onsets: Dict[str, List[int]] = {}
    track_prev_blinking: Dict[str, bool] = {}

    def _ensure_track(fid: str, cx: float, frame_idx: int) -> None:
        if fid not in track_features:
            track_centers[fid] = cx
            track_features[fid] = [
                VisionFeatures(face_detected=False, face_id_hint=fid, quality_flag="no_face")
                for _ in range(frame_idx)
            ]
            track_valid_flags[fid] = [False] * frame_idx
            track_blink_onsets[fid] = [0] * frame_idx
            track_prev_blinking[fid] = False

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for idx, raw_frame in enumerate(frames_rgb):
            if raw_frame is None or raw_frame.size == 0:
                for fid in track_features:
                    track_features[fid].append(
                        VisionFeatures(face_detected=False, face_id_hint=fid, quality_flag="empty_frame")
                    )
                    track_valid_flags[fid].append(False)
                    track_blink_onsets[fid].append(0)
                continue
            if float(np.mean(raw_frame)) < min_luma:
                for fid in track_features:
                    track_features[fid].append(
                        VisionFeatures(
                            face_detected=False,
                            face_id_hint=fid,
                            low_light_flag=True,
                            quality_flag="low_light",
                        )
                    )
                    track_valid_flags[fid].append(False)
                    track_blink_onsets[fid].append(0)
                continue

            frame = np.stack([raw_frame, raw_frame, raw_frame], axis=-1) if raw_frame.ndim == 2 else raw_frame
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            h, w = frame.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            results = landmarker.detect_for_video(mp_image, idx * step_ms)
            faces_lms = getattr(results, "face_landmarks", None) or []
            faces_bs = getattr(results, "face_blendshapes", None) or []
            faces_tm = getattr(results, "facial_transformation_matrixes", None) or []

            # Compute horizontal spatial center for each detected face
            detections: List[Tuple[float, int]] = []
            for f_i, lms in enumerate(faces_lms[:eff_max_faces]):
                cx = float(np.mean([lm.x for lm in lms])) if lms else 0.5
                detections.append((cx, f_i))

            assigned_tracks: Dict[str, int] = {}
            used_f_indices: set[int] = set()

            # Greedy nearest horizontal center assignment to existing tracks
            if track_centers and detections:
                pairs: List[Tuple[float, str, int]] = []
                for fid, t_cx in track_centers.items():
                    for cx, f_i in detections:
                        pairs.append((abs(cx - t_cx), fid, f_i))
                pairs.sort(key=lambda x: x[0])
                for dist, fid, f_i in pairs:
                    if fid in assigned_tracks or f_i in used_f_indices:
                        continue
                    if dist <= 0.22:
                        assigned_tracks[fid] = f_i
                        used_f_indices.add(f_i)

            # Create new tracks for unassigned detections if capacity remains
            for cx, f_i in detections:
                if f_i in used_f_indices:
                    continue
                if len(track_features) < eff_max_faces:
                    new_fid = f"face_{len(track_features)}"
                    _ensure_track(new_fid, cx, idx)
                    assigned_tracks[new_fid] = f_i
                    used_f_indices.add(f_i)

            for fid in list(track_features.keys()):
                if fid not in assigned_tracks:
                    track_features[fid].append(
                        VisionFeatures(face_detected=False, face_id_hint=fid, quality_flag="no_face")
                    )
                    track_valid_flags[fid].append(False)
                    track_blink_onsets[fid].append(0)
                    continue

                f_i = assigned_tracks[fid]
                lms = faces_lms[f_i]
                cx = float(np.mean([lm.x for lm in lms])) if lms else track_centers[fid]
                track_centers[fid] = 0.8 * track_centers[fid] + 0.2 * cx

                coords = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)
                blendshapes_map = (
                    {
                        str(b.category_name): float(b.score)
                        for b in faces_bs[f_i]
                        if getattr(b, "category_name", None)
                    }
                    if f_i < len(faces_bs)
                    else {}
                )
                transform_matrix = faces_tm[f_i] if f_i < len(faces_tm) else None
                vf = _build_vision_features_from_landmarks(
                    coords=coords,
                    image_width=w,
                    image_height=h,
                    ear_blink_threshold=ear_blink_threshold,
                    max_head_pose_deg=max_head_pose_deg,
                    max_pitch_deg=max_pitch_deg,
                    blendshapes_map=blendshapes_map,
                    transform_matrix=transform_matrix,
                ).model_copy(update={"face_id_hint": fid})

                is_valid = vf.quality_flag == "ok"
                onset = 0
                if is_valid:
                    is_blinking = vf.action_units.get("AU45_blink", 0.0) >= 0.5
                    if is_blinking and not track_prev_blinking[fid]:
                        onset = 1
                    track_prev_blinking[fid] = is_blinking
                track_valid_flags[fid].append(is_valid)
                track_blink_onsets[fid].append(onset)
                if is_valid:
                    lo = max(0, len(track_valid_flags[fid]) - win_frames)
                    win_valid = sum(track_valid_flags[fid][lo:])
                    win_valid_sec = float(win_valid) / eff_fps
                    if win_valid_sec >= 6.0:
                        win_blinks = sum(track_blink_onsets[fid][lo:])
                        vf = vf.model_copy(
                            update={"blink_rate_per_min": float(win_blinks) / (win_valid_sec / 60.0)}
                        )
                track_features[fid].append(vf)

    if not track_features:
        return {
            "face_0": [
                VisionFeatures(face_detected=False, face_id_hint="face_0", quality_flag="no_face")
                for _ in range(n_frames)
            ]
        }
    return track_features
