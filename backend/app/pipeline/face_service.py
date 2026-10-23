from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Generator, Optional
import uuid

import cv2
import numpy as np

from app.core.config import settings

# ---------------------------------------------------------------------------
# UniFace: High-Accuracy Biometric Face Analysis (RetinaFace + AdaFace 512-D)
# ---------------------------------------------------------------------------
try:
    from uniface import FaceAnalyzer
    from uniface.constants import AdaFaceWeights
    from uniface.recognition import AdaFace
    from uniface.recognition.base import face_alignment
    _UNIFACE_AVAILABLE = True
except ImportError:
    FaceAnalyzer = None  # type: ignore[assignment, misc]
    AdaFace = None  # type: ignore[assignment, misc]
    AdaFaceWeights = None  # type: ignore[assignment, misc]
    face_alignment = None  # type: ignore[assignment, misc]
    _UNIFACE_AVAILABLE = False

# UniFace FaceMesh: 478 3D Dense Landmarks & Iris Keypoints
try:
    from uniface import FaceMesh
    from uniface.constants import FaceMeshWeights
    from uniface.landmark import IRIS_LEFT, IRIS_RIGHT
    _FACEMESH_AVAILABLE = True
except ImportError:
    FaceMesh = None  # type: ignore[assignment, misc]
    FaceMeshWeights = None  # type: ignore[assignment, misc]
    IRIS_LEFT = None  # type: ignore[assignment, misc]
    IRIS_RIGHT = None  # type: ignore[assignment, misc]
    _FACEMESH_AVAILABLE = False

# EDifFIQA: Neural Face Image Quality Assessment
try:
    from uniface import EDifFIQA
    _EDIFFIQA_AVAILABLE = True
except ImportError:
    EDifFIQA = None  # type: ignore[assignment, misc]
    _EDIFFIQA_AVAILABLE = False

logger = logging.getLogger("SIH26187.FaceService")


def compute_3d_head_pose_angles(landmarks_3d: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Roll, Yaw, Pitch in degrees from 3D FaceMesh landmarks (478x3 or 468x3).
    Returns (roll_deg, yaw_deg, pitch_deg).
    """
    if landmarks_3d is None or len(landmarks_3d) < 468:
        return 0.0, 0.0, 0.0
    lms = landmarks_3d
    # Roll: 2D angle between outer eye corners (33 left, 263 right)
    dx_roll = float(lms[263, 0] - lms[33, 0])
    dy_roll = float(lms[263, 1] - lms[33, 1])
    roll = float(np.degrees(np.arctan2(dy_roll, max(dx_roll, 1e-4))))

    # Yaw: Horizontal depth differential between right and left face contours
    dz_yaw = float(lms[263, 2] - lms[33, 2])
    dx_yaw = float(lms[263, 0] - lms[33, 0])
    yaw = float(np.degrees(np.arctan2(dz_yaw, max(dx_yaw, 1e-4))))

    # Pitch: Vertical depth differential between top of forehead (10) and bottom of chin (152)
    dz_pitch = float(lms[152, 2] - lms[10, 2])
    dy_pitch = float(lms[152, 1] - lms[10, 1])
    pitch = float(np.degrees(np.arctan2(dz_pitch, max(dy_pitch, 1e-4))))

    return roll, yaw, pitch


def compute_eye_aspect_ratio(landmarks_3d: np.ndarray) -> tuple[float, float]:
    """
    Compute Eye Aspect Ratio (EAR) for left and right eyes using MediaPipe indices.
    Values < 0.18 typically signify blinking, squinting, or closed eyes.
    Returns (ear_left, ear_right).
    """
    if landmarks_3d is None or len(landmarks_3d) < 468:
        return 0.35, 0.35
    lms = landmarks_3d
    # Left eye: corner1=33, top1=160, top2=158, corner2=133, bot1=153, bot2=144
    v1_l = float(np.linalg.norm(lms[160, :2] - lms[144, :2]))
    v2_l = float(np.linalg.norm(lms[158, :2] - lms[153, :2]))
    h_l = float(np.linalg.norm(lms[33, :2] - lms[133, :2]))
    ear_left = (v1_l + v2_l) / max(2.0 * h_l, 1e-4)

    # Right eye: corner1=362, top1=385, top2=387, corner2=263, bot1=373, bot2=380
    v1_r = float(np.linalg.norm(lms[385, :2] - lms[380, :2]))
    v2_r = float(np.linalg.norm(lms[387, :2] - lms[373, :2]))
    h_r = float(np.linalg.norm(lms[362, :2] - lms[263, :2]))
    ear_right = (v1_r + v2_r) / max(2.0 * h_r, 1e-4)

    return float(ear_left), float(ear_right)


def compute_mouth_aspect_ratio(landmarks_3d: np.ndarray) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) to detect non-neutral facial expressions (laughing, open mouth).
    Values > 0.42 indicate wide open mouth / extreme expression.
    """
    if landmarks_3d is None or len(landmarks_3d) < 468:
        return 0.10
    lms = landmarks_3d
    # Upper lip=13, Lower lip=14, Left corner=61, Right corner=291
    v = float(np.linalg.norm(lms[13, :2] - lms[14, :2]))
    h = float(np.linalg.norm(lms[61, :2] - lms[291, :2]))
    return float(v / max(h, 1e-4))


def extract_canonical_5pts_from_mesh(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    Extract sub-pixel 5 canonical alignment keypoints [left_pupil, right_pupil, nose_tip, left_mouth, right_mouth].
    Uses iris centers if available (478 pts), falling back to eye corners (468 pts).
    """
    lms = landmarks_3d
    if len(lms) >= 478 and IRIS_LEFT is not None and IRIS_RIGHT is not None:
        left_pupil = lms[IRIS_LEFT][0, :2]
        right_pupil = lms[IRIS_RIGHT][0, :2]
    else:
        left_pupil = (lms[33, :2] + lms[133, :2]) / 2.0
        right_pupil = (lms[362, :2] + lms[263, :2]) / 2.0

    nose_tip = lms[1, :2]
    left_mouth = lms[61, :2]
    right_mouth = lms[291, :2]

    return np.array([left_pupil, right_pupil, nose_tip, left_mouth, right_mouth], dtype=np.float32)


class QualityPredictorAdapter:
    """
    Adapter bridging UniFace EDifFIQA to the FaceAnalyzer predictor interface.
    FaceAnalyzer calls predictor.predict(image, face), but EDifFIQA expects
    predict(image, landmarks). This adapter extracts 5-point landmarks, calls
    EDifFIQA, and attaches the resulting QualityResult to face.quality.
    """
    def __init__(self, ediffiqa: Any):
        self._ediffiqa = ediffiqa

    def predict(self, image: np.ndarray, face: Any) -> Any:
        try:
            landmarks = getattr(face, "landmarks", None)
            if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape[0] >= 5:
                res = self._ediffiqa.predict(image, landmarks[:5])
                face.quality = res
                return res
        except Exception as e:
            logger.debug("QualityPredictorAdapter scoring error: %s", e)
        return None


def compute_iou(boxA: list[float], boxB: list[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


def enhance_aligned_face(aligned_face: np.ndarray, is_distant_or_low_res: bool = False) -> np.ndarray:
    """
    Normalize illumination, shadows, and contrast gradients on aligned 112x112 face crop using CLAHE.
    If is_distant_or_low_res is True, also applies adaptive unsharp masking and edge sharpening
    to restore high-frequency facial cues (eyes, nose, mouth contours) lost during distant optical capture.
    Computational cost is ~0.20ms per face.
    """
    if aligned_face is None or aligned_face.size == 0:
        return aligned_face
    try:
        lab = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_eq = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)
        if is_distant_or_low_res:
            blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
            enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)
        return enhanced
    except Exception:
        return aligned_face


def deduplicate_face_detections(detections: list[dict[str, Any]], iou_thresh: float = 0.35) -> list[dict[str, Any]]:
    """Merge overlapping face detections using NMS-like IoU suppression."""
    if not detections:
        return []
    sorted_dets = sorted(
        detections,
        key=lambda d: (float(d.get("match_score", 0.0)), float(d.get("confidence", 0.0))),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for d in sorted_dets:
        boxA = d["bbox"]
        overlap = False
        for k in kept:
            boxB = k["bbox"]
            if compute_iou(boxA, boxB) >= iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(d)
    return kept


def compute_cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """
    Safe cosine similarity between two embedding vectors.
    Evaluates dot product on L2-normalized 512-D vectors in range [-1, 1].
    """
    v1 = feat1.flatten().astype(np.float32)
    v2 = feat2.flatten().astype(np.float32)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def calibrate_arcface_confidence(cosine_score: float, threshold: float = 0.32) -> float:
    """
    Calibrate ArcFace / AdaFace cosine similarity (L2-normalized, range -1 to 1)
    into an intuitive, human-friendly confidence percentage (0% to 100%).

    ArcFace/AdaFace calibration breakpoints (512-D embeddings):
      - c < 0.15: Non-match / noise         ->  0% - 12%
      - c < threshold (0.32): Below boundary -> 12% - 49%
      - c = threshold: Decision boundary     -> 50%
      - c = 0.50: Solid match               -> 85%
      - c >= 0.60: Near-certain verified     -> 95% - 99.5%
    """
    c = float(cosine_score)
    th = float(threshold)
    if c <= 0.0:
        return 0.0
    elif c <= 0.15:
        val = (c / 0.15) * 12.0
    elif c < th:
        val = 12.0 + ((c - 0.15) / max(0.01, th - 0.15)) * 37.0
    elif c <= 0.50:
        val = 50.0 + ((c - th) / max(0.01, 0.50 - th)) * 35.0
    elif c <= 0.60:
        val = 85.0 + ((c - 0.50) / 0.10) * 10.0
    else:
        val = 95.0 + min(1.0, (c - 0.60) / 0.25) * 4.5
    return round(float(np.clip(val, 0.0, 99.9)), 1)


def calibrate_match_confidence(cosine_score: float, threshold: float = 0.32) -> float:
    """Unified confidence percentage mapping using UniFace AdaFace 512-D curve."""
    return calibrate_arcface_confidence(cosine_score, threshold=threshold)


class FaceService:
    """
    Production-grade facial recognition service for MAATRIX / MayaJaal.

    Exclusive Engine: UniFace (RetinaFace / SCRFD detector + AdaFace IR-101 512-D recognizer).
    Thread-safe and optimized for real-time video streams, distant surveillance, and REST APIs.
    """

    def __init__(self) -> None:
        self.models_dir = settings.model_directory
        self.data_dir = settings.model_directory.parent / "app" / "data" / "faces"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.db_file = self.data_dir / "known_faces.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

        # --- UniFace engine (Exclusive) ---
        self.analyzer: Optional[Any] = None
        self.mesher: Optional[Any] = None
        self.quality_scorer: Optional[Any] = None
        self.use_uniface: bool = True
        self.embedding_dim: int = 512
        self.match_threshold: float = 0.32

        self.lock = threading.Lock()

        # In-memory known persons list
        self.known_persons: list[dict[str, Any]] = []

        # Recent detection logs for live operational dashboard
        self.recent_events: list[dict[str, Any]] = []
        self.max_events: int = 50

        # Temporal tracking history to eliminate video jitter and momentary false alerts
        self.tracks: dict[int, dict[str, Any]] = {}
        self.next_track_id: int = 1

        # Number of consecutive Unknown frames before a confirmed-identity track releases its lock.
        # Increase this value for more stability; decrease for faster identity release on occlusion.
        # Default=8 at ~30fps with face-detection every 4th frame ≈ ≥ 1 second of bridging.
        self.identity_drop_tolerance: int = 8

        # Minimum number of known votes required to confirm and lock an identity on a track.
        # 1 = lock fires immediately on the first confident known frame (recommended for stability).
        self.identity_lock_threshold: int = 1

        self._init_models()
        self.load_database()

    def _init_models(self) -> None:
        """Initialize the exclusive UniFace (RetinaFace/SCRFD + AdaFace 512-D) engine."""
        if not _UNIFACE_AVAILABLE:
            logger.error("UniFace package is not available. Please install uniface via pip.")
            return

        try:
            # Initialize EDifFIQA quality scorer if available
            quality_predictors = []
            if _EDIFFIQA_AVAILABLE and EDifFIQA is not None:
                try:
                    self.quality_scorer = EDifFIQA()
                    quality_predictors.append(QualityPredictorAdapter(self.quality_scorer))
                    logger.info("EDifFIQA face quality scorer loaded with FaceAnalyzer adapter.")
                except Exception as q_err:
                    logger.warning("EDifFIQA initialization failed (non-critical): %s", q_err)

            # Initialize FaceMesh V2_478 for high-precision enrollment & alignment
            if _FACEMESH_AVAILABLE and FaceMesh is not None and FaceMeshWeights is not None:
                try:
                    self.mesher = FaceMesh(model_name=FaceMeshWeights.V2_478)
                    logger.info("FaceMesh (V2_478 3D + irises) loaded for biometric enrollment and alignment.")
                except Exception as m_err:
                    logger.warning("FaceMesh initialization failed (fallback to detector landmarks): %s", m_err)

            self.analyzer = FaceAnalyzer(
                recognizer=AdaFace(model_name=AdaFaceWeights.IR_101),
                predictors=quality_predictors if quality_predictors else None,
            )
            self.use_uniface = True
            self.embedding_dim = 512
            self.match_threshold = 0.32
            features_tag = []
            if self.mesher:
                features_tag.append("FaceMesh V2_478")
            if self.quality_scorer:
                features_tag.append("EDifFIQA")
            tag_str = f" + {' + '.join(features_tag)}" if features_tag else ""
            logger.info("FaceService initialized with UniFace (AdaFace IR-101, 512-D embeddings%s).", tag_str)
        except Exception as exc:
            logger.exception("Failed to initialize UniFace FaceAnalyzer: %s", exc)

    def _score_face_quality(
        self,
        face_obj: Any,
        image_bgr: Optional[np.ndarray] = None,
        aligned_crop: Optional[np.ndarray] = None,
    ) -> float:
        """
        Extract EDifFIQA quality score from a UniFace Face object or aligned face crop.
        Returns a score in [0.0, 1.0] where higher = better quality.
        Falls back to 0.5 (neutral) if quality scoring is unavailable.
        """
        if face_obj is not None:
            quality = getattr(face_obj, "quality", None)
            if quality is not None:
                if hasattr(quality, "score"):
                    return float(np.clip(quality.score, 0.0, 1.0))
                try:
                    return float(np.clip(float(quality), 0.0, 1.0))
                except (TypeError, ValueError):
                    pass

        # Fallback 1: score aligned crop directly if provided
        if aligned_crop is not None and self.quality_scorer is not None:
            try:
                res = self.quality_scorer.score_aligned(aligned_crop)
                score = res.score if hasattr(res, "score") else float(res)
                if face_obj is not None:
                    face_obj.quality = res
                return float(np.clip(score, 0.0, 1.0))
            except Exception as e:
                logger.debug("score_aligned failed: %s", e)

        # Fallback 2: score using image_bgr and face landmarks
        if (
            face_obj is not None
            and image_bgr is not None
            and self.quality_scorer is not None
        ):
            landmarks = getattr(face_obj, "landmarks", None)
            if landmarks is not None and isinstance(landmarks, np.ndarray) and landmarks.shape[0] >= 5:
                try:
                    res = self.quality_scorer.predict(image_bgr, landmarks[:5])
                    score = res.score if hasattr(res, "score") else float(res)
                    face_obj.quality = res
                    return float(np.clip(score, 0.0, 1.0))
                except Exception as e:
                    logger.debug("EDifFIQA.predict failed: %s", e)

        return 0.5

    @property
    def engine_name(self) -> str:
        """Human-readable active engine identifier for status endpoints."""
        parts = ["AdaFace IR-101 512-D"]
        if self.mesher is not None:
            parts.append("FaceMesh V2_478")
        if self.quality_scorer is not None:
            parts.append("EDifFIQA")
        return f"UniFace ({' + '.join(parts)})"

    @property
    def is_ready(self) -> bool:
        """True when the UniFace engine is operational."""
        return self.analyzer is not None

    @property
    def detector(self) -> Any:
        """Backwards compatibility accessor for underlying detector."""
        return self.analyzer.detector if self.analyzer else None

    @property
    def recognizer(self) -> Any:
        """Backwards compatibility accessor for underlying recognizer."""
        return self.analyzer.recognizer if self.analyzer else None

    @property
    def yunet_path(self) -> Path:
        """Backwards compatibility property."""
        return Path("N/A")

    @property
    def sface_path(self) -> Path:
        """Backwards compatibility property."""
        return Path("N/A")

    @staticmethod
    def _compute_centroid(
        features_np: list[np.ndarray],
        quality_weights: Optional[list[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute the L2-normalized mean embedding (centroid) across all enrolled 512-D samples.
        If quality_weights are provided, uses quality-weighted averaging so that clear,
        high-quality enrollment photos contribute more to the identity anchor than blurry ones.
        """
        if not features_np:
            return None
        valid_features = [f.reshape(1, -1) for f in features_np if f.flatten().shape[0] == 512]
        if not valid_features:
            return None
        stacked = np.vstack(valid_features).astype(np.float32)

        if quality_weights and len(quality_weights) == len(valid_features):
            # Quality-weighted centroid: high-quality samples dominate
            weights = np.array(quality_weights, dtype=np.float32).reshape(-1, 1)
            # Floor at 0.3 so even low-quality samples contribute slightly
            weights = np.clip(weights, 0.3, 1.0)
            weights = weights / weights.sum()
            mean_vec = (weights * stacked).sum(axis=0, keepdims=True)
        else:
            mean_vec = np.mean(stacked, axis=0, keepdims=True)

        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            mean_vec = (mean_vec / norm).astype(np.float32)
        return mean_vec

    def extract_tta_embedding(
        self,
        image_bgr: np.ndarray,
        landmarks: np.ndarray,
        is_distant: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extract a Test-Time Augmented (TTA) 512-D embedding directly from native resolution.
        Aligns the face with similarity transform, then evaluates both original aligned crop
        and its horizontal mirror reflection to cancel out lighting and pose asymmetry.
        If is_distant is True, applies adaptive unsharp masking to enhance blurred features.
        """
        if self.analyzer is None or self.analyzer.recognizer is None or face_alignment is None:
            return None
        rec = self.analyzer.recognizer
        try:
            aligned_face, _ = face_alignment(
                image_bgr, landmarks, image_size=rec.input_size
            )
            if aligned_face is None or aligned_face.size == 0:
                return None

            # Always normalize illumination using CLAHE; apply unsharp masking if distant/low-res
            aligned_face = enhance_aligned_face(aligned_face, is_distant_or_low_res=is_distant)

            blob1 = rec.preprocess(aligned_face)
            e1 = rec.session.run(rec.output_names, {rec.input_name: blob1})[0].ravel()

            flipped_face = cv2.flip(aligned_face, 1)
            blob2 = rec.preprocess(flipped_face)
            e2 = rec.session.run(rec.output_names, {rec.input_name: blob2})[0].ravel()

            combined = (e1 + e2).astype(np.float32)
            norm = np.linalg.norm(combined)
            if norm > 1e-8:
                emb = (combined / norm).reshape(1, 512)
            else:
                emb = e1.astype(np.float32).reshape(1, 512)
            return emb
        except Exception as exc:
            logger.debug("TTA embedding extraction failed: %s", exc)
            return None

    def extract_face_crop_embedding(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a normalized 512-D AdaFace embedding directly from a face crop.
        Used for Unknown Person Re-ID, temporal tracking, and multi-camera face clustering.
        """
        if face_bgr is None or face_bgr.size == 0 or self.analyzer is None:
            return None
        try:
            # 1. Try detector + aligner on the crop
            with self.lock:
                faces = self.analyzer.analyze(face_bgr)
            if faces and faces[0].embedding is not None:
                emb = faces[0].embedding.astype(np.float32).flatten()
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    return (emb / norm).astype(np.float32)
                return emb.astype(np.float32)

            # 2. If crop is tight and detector missed, run recognizer directly on resized crop
            if self.analyzer.recognizer is not None:
                rec = self.analyzer.recognizer
                in_sz = tuple(rec.input_size)
                resized = cv2.resize(face_bgr, in_sz, interpolation=cv2.INTER_LANCZOS4)
                enhanced = enhance_aligned_face(resized, is_distant_or_low_res=False)
                blob = rec.preprocess(enhanced)
                feat = rec.session.run(rec.output_names, {rec.input_name: blob})[0].ravel().astype(np.float32)
                norm = np.linalg.norm(feat)
                if norm > 1e-8:
                    return (feat / norm).astype(np.float32)
                return feat
        except Exception as exc:
            logger.debug("Failed to extract face crop embedding: %s", exc)
        return None

    def load_database(self) -> None:
        """
        Load enrolled persons into memory (512-D UniFace).
        Auto-heals any legacy 128-D profiles using stored thumbnail images.
        """
        with self.lock:
            self.known_persons = []
            if not self.db_file.exists():
                with open(self.db_file, "w", encoding="utf-8") as f:
                    json.dump([], f)
                return

            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                db_modified = False
                for r in records:
                    features_np: list[np.ndarray] = []
                    raw_features = r.get("features", [])
                    if isinstance(raw_features, list) and len(raw_features) > 0:
                        for item in raw_features:
                            if isinstance(item, list) and len(item) == 512:
                                features_np.append(
                                    np.array(item, dtype=np.float32).reshape(1, 512)
                                )

                    single_feat = r.get("feature", [])
                    if len(features_np) == 0 and isinstance(single_feat, list) and len(single_feat) == 512:
                        features_np.append(np.array(single_feat, dtype=np.float32).reshape(1, 512))

                    person_id = r.get("id")
                    thumb_name = f"{person_id}.jpg"
                    image_url = f"/api/faces/thumbnail/{thumb_name}"
                    is_suspect = bool(r.get("is_suspect", False))
                    threat_level = str(r.get("threat_level", "HIGH" if is_suspect else "LOW"))
                    category = str(r.get("category", "Wanted / BOLO" if is_suspect else "Authorized Personnel"))
                    notes = str(r.get("notes", ""))

                    # Auto-heal legacy 128-D entries to 512-D UniFace from thumbnail
                    thumb_file = self.thumbnails_dir / thumb_name
                    if (not features_np or features_np[0].shape[1] != 512) and thumb_file.exists() and self.analyzer:
                        try:
                            t_img = cv2.imread(str(thumb_file))
                            if t_img is not None:
                                t_faces = self.analyzer.analyze(t_img)
                                if t_faces:
                                    t_face = max(t_faces, key=lambda f: float(f.confidence))
                                    emb = None
                                    if hasattr(t_face, "landmarks") and t_face.landmarks is not None:
                                        emb = self.extract_tta_embedding(t_img, t_face.landmarks)
                                    if emb is None and t_face.embedding is not None:
                                        emb = t_face.embedding.astype(np.float32).reshape(1, 512)
                                    if emb is not None:
                                        features_np = [emb]
                                        r["features"] = [emb.tolist()[0]]
                                        r["feature"] = emb.tolist()[0]
                                        db_modified = True
                                        logger.info("Auto-migrated %s to 512-D UniFace from thumbnail.", r.get("name"))
                        except Exception as heal_err:
                            logger.debug("Auto-heal skipped for %s: %s", r.get("name"), heal_err)

                    if not features_np:
                        continue

                    centroid = self._compute_centroid(features_np)

                    self.known_persons.append({
                        "id": person_id,
                        "name": r.get("name"),
                        "features": features_np,
                        "feature": features_np[0],
                        "centroid": centroid,
                        "image_url": image_url,
                        "created_at": r.get("created_at"),
                        "is_suspect": is_suspect,
                        "threat_level": threat_level,
                        "category": category,
                        "notes": notes,
                    })

                if db_modified:
                    self._save_database()

                logger.info("Loaded %d enrolled person(s) into UniFace database (512-D).", len(self.known_persons))
            except Exception as exc:
                logger.exception("Error loading known_faces.json: %s", exc)

    def _save_database(self) -> None:
        """Persist enrolled persons to known_faces.json."""
        records = []
        for p in self.known_persons:
            records.append({
                "id": p["id"],
                "name": p["name"],
                "image_url": p["image_url"],
                "created_at": p["created_at"],
                "features": [f.tolist()[0] for f in p.get("features", [p["feature"]])],
                "feature": p["feature"].tolist()[0],
                "is_suspect": p.get("is_suspect", False),
                "threat_level": p.get("threat_level", "HIGH" if p.get("is_suspect") else "LOW"),
                "category": p.get("category", "Wanted / BOLO" if p.get("is_suspect") else "Authorized Personnel"),
                "notes": p.get("notes", ""),
            })
        try:
            with open(self.db_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception as exc:
            logger.error("Failed to save face database: %s", exc)

    def list_persons(self) -> list[dict[str, Any]]:
        """Return a public metadata list of all enrolled persons."""
        with self.lock:
            return [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "image_url": p["image_url"],
                    "created_at": p["created_at"],
                    "sample_count": len(p.get("features", [1])),
                    "is_suspect": p.get("is_suspect", False),
                    "threat_level": p.get("threat_level", "HIGH" if p.get("is_suspect") else "LOW"),
                    "category": p.get("category", "Wanted / BOLO" if p.get("is_suspect") else "Authorized Personnel"),
                    "notes": p.get("notes", ""),
                }
                for p in self.known_persons
            ]

    def get_person(self, person_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single enrolled person by ID."""
        with self.lock:
            p = next((x for x in self.known_persons if x["id"] == person_id), None)
            if not p:
                return None
            return {
                "id": p["id"],
                "name": p["name"],
                "image_url": p["image_url"],
                "created_at": p["created_at"],
                "sample_count": len(p.get("features", [1])),
                "is_suspect": p.get("is_suspect", False),
                "threat_level": p.get("threat_level", "HIGH" if p.get("is_suspect") else "LOW"),
                "category": p.get("category", "Wanted / BOLO" if p.get("is_suspect") else "Authorized Personnel"),
                "notes": p.get("notes", ""),
            }

    def update_classification(
        self,
        person_id: str,
        is_suspect: Optional[bool] = None,
        threat_level: Optional[str] = None,
        category: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update classification and threat metadata for an enrolled person."""
        with self.lock:
            person = next((p for p in self.known_persons if p["id"] == person_id), None)
            if not person:
                return {"success": False, "error": "Person ID not found"}

            if is_suspect is not None:
                person["is_suspect"] = bool(is_suspect)
            if threat_level is not None:
                person["threat_level"] = str(threat_level).upper()
            if category is not None:
                person["category"] = str(category).strip()
            if notes is not None:
                person["notes"] = str(notes).strip()

            self._save_database()
            return {
                "success": True,
                "person": {
                    "id": person["id"],
                    "name": person["name"],
                    "image_url": person["image_url"],
                    "created_at": person["created_at"],
                    "sample_count": len(person.get("features", [1])),
                    "is_suspect": person.get("is_suspect", False),
                    "threat_level": person.get("threat_level", "HIGH" if person.get("is_suspect") else "LOW"),
                    "category": person.get("category", "Wanted / BOLO" if person.get("is_suspect") else "Authorized Personnel"),
                    "notes": person.get("notes", ""),
                },
                "message": f"Updated classification for {person['name']}",
            }

    update_person_metadata = update_classification

    def register_face(
        self,
        image_bgr: np.ndarray,
        name: str,
        is_suspect: bool = True,
        threat_level: str = "HIGH",
        category: str = "Wanted / BOLO",
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Enroll a new person or add a sample to an existing profile using UniFace (512-D).
        Extracts high-res native TTA embedding + synthetic multi-scale distance anchors.
        """
        name = name.strip()
        if not name:
            return {"success": False, "error": "Name cannot be blank"}
        if image_bgr is None or image_bgr.size == 0:
            return {"success": False, "error": "Invalid image data"}
        if not self.is_ready or self.analyzer is None:
            return {"success": False, "error": "UniFace recognition engine is not initialized"}

        try:
            with self.lock:
                faces_uf = self.analyzer.analyze(image_bgr)

            # Fallback for tightly-cropped registration images where detector may miss
            if not faces_uf and self.mesher is not None:
                h_img, w_img = image_bgr.shape[:2]
                if w_img >= 40 and h_img >= 40:
                    try:
                        probe_res = self.mesher.predict(image_bgr, bboxes=[[0, 0, w_img, h_img]])
                        if probe_res and float(getattr(probe_res[0], "score", 0.0)) >= 0.85:
                            class _SyntheticFace:
                                def __init__(self, bbox: list[float], conf: float):
                                    self.bbox = bbox
                                    self.confidence = conf
                                    self.landmarks = None
                                    self.embedding = None
                            faces_uf = [_SyntheticFace([0.0, 0.0, float(w_img), float(h_img)], float(probe_res[0].score))]
                    except Exception as probe_err:
                        logger.debug("FaceMesh probe fallback error: %s", probe_err)

            if not faces_uf:
                return {
                    "success": False,
                    "error": "No face detected in this image. Please ensure good lighting and face the camera directly.",
                }

            best_uf = max(faces_uf, key=lambda f: float(f.confidence))
            bbox_arr = best_uf.bbox
            fw = float(bbox_arr[2] - bbox_arr[0])
            fh = float(bbox_arr[3] - bbox_arr[1])
            det_conf = float(best_uf.confidence)

            if fw < 40 or fh < 40:
                return {
                    "success": False,
                    "error": f"Face is too small ({int(fw)}x{int(fh)} px). Minimum 40x40 px required for registration.",
                }

            if det_conf < 0.40:
                return {
                    "success": False,
                    "error": f"Face detection confidence is low ({int(det_conf * 100)}%). Please face the camera directly in clear light.",
                }

            # EDifFIQA Neural Quality Gate: reject blurry/occluded/poorly-lit faces
            enrollment_quality = self._score_face_quality(best_uf, image_bgr=image_bgr)
            if enrollment_quality < 0.25 and self.quality_scorer is not None:
                return {
                    "success": False,
                    "error": f"Face image quality is too low ({int(enrollment_quality * 100)}%). "
                             f"Please ensure clear lighting, no blur, and face the camera directly.",
                }
            if enrollment_quality < 0.25:
                logger.warning("Low enrollment quality (%.2f) for %s but EDifFIQA not available — proceeding.", enrollment_quality, name)

            # -------------------------------------------------------------
            # FACEMESH DENSE 3D + IRIS VERIFICATION & SUB-PIXEL ALIGNMENT
            # -------------------------------------------------------------
            mesh_verified = False
            canonical_5pts = None
            mesh_metrics: dict[str, Any] = {}

            if self.mesher is not None:
                try:
                    bx1 = max(0.0, float(bbox_arr[0]))
                    by1 = max(0.0, float(bbox_arr[1]))
                    bx2 = min(float(image_bgr.shape[1]), float(bbox_arr[2]))
                    by2 = min(float(image_bgr.shape[0]), float(bbox_arr[3]))
                    mesh_results = self.mesher.predict(image_bgr, bboxes=[[bx1, by1, bx2, by2]])

                    if mesh_results:
                        res = mesh_results[0]
                        presence_score = float(getattr(res, "score", 1.0))
                        lms_3d = res.landmarks

                        if presence_score < 0.70:
                            return {
                                "success": False,
                                "error": f"Facial mesh presence confidence too low ({int(presence_score * 100)}%). "
                                         f"Please remove face coverings and ensure clear illumination.",
                            }

                        # Check 1: Eye Aspect Ratio (Blink / Squint Gate)
                        ear_left, ear_right = compute_eye_aspect_ratio(lms_3d)
                        if ear_left < 0.18 or ear_right < 0.18:
                            return {
                                "success": False,
                                "error": "Eyes appear closed or squinting during enrollment. "
                                         "Please look directly into the camera with both eyes open for optimal recognition.",
                            }

                        # Check 2: Mouth Aspect Ratio (Facial Expression Gate)
                        mar = compute_mouth_aspect_ratio(lms_3d)
                        if mar > 0.42:
                            return {
                                "success": False,
                                "error": "Non-neutral facial expression or open mouth detected. "
                                         "Please keep a relaxed, neutral expression during enrollment.",
                            }

                        # Check 3: True 3D Head Pose Angles (Allows Quarter Profiles up to 35° Yaw)
                        roll_deg, yaw_deg, pitch_deg = compute_3d_head_pose_angles(lms_3d)

                        if abs(roll_deg) > 18.0:
                            return {
                                "success": False,
                                "error": f"Head is tilted ({int(roll_deg)}°). Please keep your head upright during enrollment.",
                            }

                        if abs(pitch_deg) > 18.0:
                            return {
                                "success": False,
                                "error": f"Head is tilted up or down (Pitch: {int(pitch_deg)}°). Maximum allowed tilt is 18°. Please look towards the camera.",
                            }

                        # Pose classification: Frontal (<= 18°) vs Quarter Profile (18° - 35°) vs Extreme Profile (> 35°)
                        if abs(yaw_deg) <= 18.0:
                            pose_type = "FRONTAL"
                        elif abs(yaw_deg) <= 35.0:
                            pose_type = "QUARTER_PROFILE_RIGHT" if yaw_deg > 0 else "QUARTER_PROFILE_LEFT"
                            # Ensure both eyes remain visible for quarter profile alignment
                            if ear_left < 0.15 or ear_right < 0.15:
                                return {
                                    "success": False,
                                    "error": "One eye is partially occluded by the head angle. "
                                             "Please turn slightly more towards the camera so both eyes are clearly visible.",
                                }
                        else:
                            return {
                                "success": False,
                                "error": f"Face angle is too steep (Yaw: {int(abs(yaw_deg))}°). "
                                         f"Maximum supported angle is 35° (Quarter Profile). "
                                         f"Please face more towards the camera so both eyes are visible.",
                            }

                        # Check 4: Extract sub-pixel canonical 5-point alignment keypoints from pupil centers
                        canonical_5pts = extract_canonical_5pts_from_mesh(lms_3d)
                        mesh_verified = True
                        mesh_metrics = {
                            "presence_score": round(presence_score, 3),
                            "pose_type": pose_type,
                            "yaw_deg": round(yaw_deg, 1),
                            "pitch_deg": round(pitch_deg, 1),
                            "roll_deg": round(roll_deg, 1),
                            "ear_left": round(ear_left, 3),
                            "ear_right": round(ear_right, 3),
                            "mar": round(mar, 3),
                        }
                        logger.info("FaceMesh enrollment gate passed: %s", mesh_metrics)

                except Exception as m_gate_err:
                    logger.warning("FaceMesh enrollment gate encountered error (falling back to 2D checks): %s", m_gate_err)

            # Fallback to 2D heuristics if FaceMesh was unavailable or skipped
            if not mesh_verified:
                if hasattr(best_uf, "landmarks") and best_uf.landmarks is not None:
                    lms = np.array(best_uf.landmarks, dtype=np.float32)
                    if len(lms) >= 5:
                        canonical_5pts = lms[:5]
                        left_eye, right_eye, nose = lms[0], lms[1], lms[2]
                        # Check Roll (head tilt angle)
                        dx = float(right_eye[0] - left_eye[0])
                        dy = float(right_eye[1] - left_eye[1])
                        roll_deg = abs(float(np.degrees(np.arctan2(dy, dx))))
                        if roll_deg > 18.0:
                            return {
                                "success": False,
                                "error": f"Head is tilted ({int(roll_deg)}°). Please keep your head upright during enrollment.",
                            }
                        # Check Yaw (allows quarter profile in 2D)
                        dist_l = float(np.linalg.norm(nose - left_eye))
                        dist_r = float(np.linalg.norm(nose - right_eye))
                        yaw_ratio = dist_l / max(dist_r, 1e-4)
                        if yaw_ratio < 0.25 or yaw_ratio > 4.0:
                            return {
                                "success": False,
                                "error": "Face angle is too steep. Please look more towards the camera.",
                            }
                        # Check Pitch
                        left_mouth, right_mouth = lms[3], lms[4]
                        eye_center = (left_eye + right_eye) / 2.0
                        mouth_center = (left_mouth + right_mouth) / 2.0
                        eye_to_nose = float(np.linalg.norm(nose - eye_center))
                        nose_to_mouth = float(np.linalg.norm(mouth_center - nose))
                        pitch_ratio = eye_to_nose / max(nose_to_mouth, 1e-4)
                        if pitch_ratio < 0.30 or pitch_ratio > 3.3:
                            return {
                                "success": False,
                                "error": "Head is tilted up or down. Please look straight at the camera during enrollment.",
                            }

            # Extract 512-D TTA embedding using sub-pixel pupil-aligned canonical keypoints
            feature_512 = None
            if canonical_5pts is not None:
                feature_512 = self.extract_tta_embedding(image_bgr, canonical_5pts)
            elif hasattr(best_uf, "landmarks") and best_uf.landmarks is not None:
                feature_512 = self.extract_tta_embedding(image_bgr, best_uf.landmarks)
            if feature_512 is None and best_uf.embedding is not None:
                feature_512 = best_uf.embedding.astype(np.float32).reshape(1, 512)

            if feature_512 is None:
                return {"success": False, "error": "Failed to extract 512-D facial feature embedding."}

            features_to_store = [feature_512]
            alignment_pts = canonical_5pts if canonical_5pts is not None else getattr(best_uf, "landmarks", None)
            if alignment_pts is not None and face_alignment is not None:
                try:
                    aligned_face, _ = face_alignment(
                        image_bgr, alignment_pts, image_size=self.analyzer.recognizer.input_size
                    )
                    if aligned_face is not None:
                        rec = self.analyzer.recognizer
                        # Generate multi-scale synthetic distance anchors to cover
                        # the full range of real-world quality degradation (40px, 56px, 80px)
                        for anchor_sz in (40, 56, 80):
                            try:
                                down = cv2.resize(aligned_face, (anchor_sz, anchor_sz), interpolation=cv2.INTER_AREA)
                                up = cv2.resize(down, (112, 112), interpolation=cv2.INTER_CUBIC)
                                b_dist = rec.preprocess(enhance_aligned_face(up, is_distant_or_low_res=True))
                                e_dist = rec.session.run(rec.output_names, {rec.input_name: b_dist})[0].ravel()
                                norm_d = np.linalg.norm(e_dist)
                                if norm_d > 1e-8:
                                    features_to_store.append((e_dist / norm_d).astype(np.float32).reshape(1, 512))
                            except Exception:
                                pass
                except Exception as dist_err:
                    logger.debug("Synthetic distance anchor generation skip: %s", dist_err)

        except Exception as exc:
            logger.exception("Error during UniFace registration: %s", exc)
            return {"success": False, "error": f"UniFace processing failed: {exc}"}

        with self.lock:
            existing = next((p for p in self.known_persons if p["name"].lower() == name.lower()), None)

            if existing:
                if "features" not in existing:
                    existing["features"] = [existing["feature"]]
                if "quality_weights" not in existing:
                    # Back-fill neutral quality weights for legacy samples
                    existing["quality_weights"] = [0.5] * len(existing["features"])

                # Side views (yaw > 18°) get a calibrated weight (0.55) so they
                # serve as dedicated angled matching targets without distorting the primary frontal centroid anchor
                is_side_view = abs(mesh_metrics.get("yaw_deg", 0.0)) > 18.0
                primary_weight = enrollment_quality * (0.55 if is_side_view else 1.0)
                sample_qualities = [primary_weight] + [0.35] * (len(features_to_store) - 1)

                for i, f_add in enumerate(features_to_store):
                    q_w = sample_qualities[i] if i < len(sample_qualities) else 0.35
                    if len(existing["features"]) < 10:
                        existing["features"].append(f_add)
                        existing["quality_weights"].append(q_w)
                    else:
                        existing["features"][0] = f_add
                        existing["quality_weights"][0] = q_w

                existing["is_suspect"] = bool(is_suspect)
                existing["threat_level"] = str(threat_level).upper()
                existing["category"] = str(category).strip() or existing.get("category", "Wanted / BOLO")
                if notes:
                    existing["notes"] = str(notes).strip()

                existing["centroid"] = self._compute_centroid(
                    existing["features"], quality_weights=existing.get("quality_weights")
                )
                self._save_database()
                pose_desc = f", {mesh_metrics['pose_type']}" if mesh_metrics.get("pose_type") and mesh_metrics["pose_type"] != "FRONTAL" else ""
                return {
                    "success": True,
                    "person_id": existing["id"],
                    "name": existing["name"],
                    "is_new": False,
                    "sample_count": len(existing["features"]),
                    "image_url": existing["image_url"],
                    "is_suspect": existing["is_suspect"],
                    "threat_level": existing["threat_level"],
                    "category": existing["category"],
                    "enrollment_quality": round(enrollment_quality, 3),
                    "mesh_verified": mesh_verified,
                    "mesh_metrics": mesh_metrics,
                    "message": f"Added samples to {name} ({len(existing['features'])} total, 512-D{pose_desc}, quality={enrollment_quality:.0%}).",
                }

            person_id = f"person_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            thumb_filename = f"{person_id}.jpg"
            thumb_path = self.thumbnails_dir / thumb_filename

            bx1 = max(0, int(bbox_arr[0]))
            by1 = max(0, int(bbox_arr[1]))
            bx2 = min(image_bgr.shape[1], int(bbox_arr[2]))
            by2 = min(image_bgr.shape[0], int(bbox_arr[3]))
            crop = image_bgr[by1:by2, bx1:bx2]
            if crop is not None and crop.size > 0:
                cv2.imwrite(str(thumb_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(str(thumb_path), image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            image_url = f"/api/faces/thumbnail/{thumb_filename}"
            created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            centroid = self._compute_centroid(
                features_to_store,
                quality_weights=[enrollment_quality] + [0.4] * (len(features_to_store) - 1),
            )

            pose_desc = f", {mesh_metrics['pose_type']}" if mesh_metrics.get("pose_type") and mesh_metrics["pose_type"] != "FRONTAL" else ""
            new_record = {
                "id": person_id,
                "name": name,
                "features": features_to_store,
                "feature": feature_512,
                "centroid": centroid,
                "quality_weights": [enrollment_quality] + [0.4] * (len(features_to_store) - 1),
                "image_url": image_url,
                "created_at": created_at,
                "is_suspect": bool(is_suspect),
                "threat_level": str(threat_level).upper(),
                "category": str(category).strip() or ("Wanted / BOLO" if is_suspect else "Authorized Personnel"),
                "notes": str(notes).strip(),
                "pose_type": mesh_metrics.get("pose_type", "FRONTAL"),
            }
            self.known_persons.append(new_record)
            self._save_database()

            mesh_str = " + FaceMesh 3D" if mesh_verified else ""
            logger.info("Enrolled %s (ID: %s, 512-D UniFace%s%s, quality=%.2f).", name, person_id, mesh_str, pose_desc, enrollment_quality)
            return {
                "success": True,
                "person_id": person_id,
                "name": name,
                "is_new": True,
                "sample_count": len(features_to_store),
                "image_url": image_url,
                "is_suspect": new_record["is_suspect"],
                "threat_level": new_record["threat_level"],
                "category": new_record["category"],
                "enrollment_quality": round(enrollment_quality, 3),
                "mesh_verified": mesh_verified,
                "mesh_metrics": mesh_metrics,
                "message": f"Successfully enrolled {name} (512-D UniFace{mesh_str}{pose_desc}, quality={enrollment_quality:.0%}).",
            }

    def add_sample(
        self,
        person_id: str,
        image_bgr: np.ndarray,
        is_suspect: Optional[bool] = None,
        threat_level: Optional[str] = None,
        category: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Add an additional training sample to an enrolled person's gallery profile."""
        with self.lock:
            person = next((p for p in self.known_persons if p["id"] == person_id), None)
            if not person:
                return {"success": False, "error": "Person ID not found"}
            name = person["name"]

        return self.register_face(
            image_bgr=image_bgr,
            name=name,
            is_suspect=is_suspect if is_suspect is not None else person.get("is_suspect", False),
            threat_level=threat_level or person.get("threat_level", "HIGH"),
            category=category or person.get("category", "Wanted / BOLO"),
            notes=notes if notes is not None else person.get("notes", ""),
        )

    def delete_person(self, person_id: str) -> dict[str, Any]:
        """Delete an enrolled person and remove their thumbnail."""
        with self.lock:
            prev_len = len(self.known_persons)
            self.known_persons = [p for p in self.known_persons if p["id"] != person_id]
            if len(self.known_persons) < prev_len:
                self._save_database()
                thumb_path = self.thumbnails_dir / f"{person_id}.jpg"
                if thumb_path.exists():
                    try:
                        thumb_path.unlink()
                    except Exception:
                        pass
                return {"success": True, "message": "Person removed from face database"}
            return {"success": False, "error": "Person ID not found"}

    def _match_embedding(
        self,
        query_feat: np.ndarray,
        known_list: list[dict[str, Any]],
        min_match_score: float,
        is_distant: bool = False,
        face_size: float = 0.0,
        probe_quality: float = 0.5,
    ) -> tuple[str, float, bool, str, str, bool]:
        """
        Match a 512-D query embedding against all enrolled persons.
        Evaluates both precomputed identity centroids and stored sample vectors.
        Uses distance-adaptive and quality-adaptive decision thresholding
        with centroid corroboration boost and smart margin guards.
        Returns (name, best_score, is_known, threat_level, category, is_ambiguous).
        """
        best_name = "Unknown"
        best_score = -1.0
        second_best_score = -1.0
        is_known = False
        best_threat_level = "LOW"
        best_category = "Unknown"

        # Quality-adaptive and distance-adaptive thresholding
        base_thresh = float(min_match_score)

        # Distant surveillance threshold adaptation:
        # Prevent false positives in 512-D AdaFace space:
        # Floor must never drop below 0.30 (random strangers regularly score 0.20-0.28).
        if is_distant:
            effective_thresh = max(0.30, base_thresh * 0.90)
        elif 0 < face_size < 80:
            effective_thresh = max(0.30, base_thresh * 0.94)
        else:
            effective_thresh = base_thresh

        # EDifFIQA Probe quality gating:
        # When probe face quality is poor/blurry (<0.28), embeddings are noisy.
        # Strict biometric rule: Do NOT lower threshold on degraded probes.
        # Instead, enforce base threshold (or stricter) to prevent background collisions.
        if probe_quality < 0.28:
            effective_thresh = max(effective_thresh, base_thresh)
        elif probe_quality >= 0.60 and is_distant:
            # High-fidelity distant crops have trustworthy features; allow slight adaptive relaxation down to 0.29 floor
            effective_thresh = max(0.29, effective_thresh * 0.96)

        if not known_list:
            return best_name, best_score, is_known, best_threat_level, best_category, False

        scored_candidates: list[tuple[dict[str, Any], float]] = []
        for p in known_list:
            p_best_score = -1.0
            centroid_score = -1.0

            centroid = p.get("centroid")
            if centroid is not None and centroid.flatten().shape[0] == 512:
                cos_c = compute_cosine_similarity(query_feat, centroid)
                centroid_score = cos_c
                if cos_c > p_best_score:
                    p_best_score = cos_c

            p_features = p.get("features") or ([p["feature"]] if "feature" in p else [])
            for k_feat in p_features:
                if k_feat.flatten().shape[0] != 512:
                    continue
                cos = compute_cosine_similarity(query_feat, k_feat)
                if cos > p_best_score:
                    p_best_score = cos

            # Centroid corroboration boost: when the centroid (stable identity
            # anchor) and the best individual sample both agree that this is a
            # near-threshold match, add a small reliability bonus.
            if centroid_score > 0 and p_best_score > 0:
                centroid_gap = p_best_score - centroid_score
                if centroid_gap < 0.05 and centroid_score >= (effective_thresh * 0.95):
                    p_best_score = min(p_best_score + 0.02, 1.0)

            if p_best_score > -1.0:
                scored_candidates.append((p, p_best_score))

        if not scored_candidates:
            return best_name, best_score, is_known, best_threat_level, best_category, False

        # Sort candidates deterministically by descending cosine score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_person, best_score = scored_candidates[0]
        second_best_score = scored_candidates[1][1] if len(scored_candidates) > 1 else -1.0

        # Margin check & Single-gallery open-set protection:
        margin_ok = True
        if len(scored_candidates) >= 2:
            margin_ok = (best_score - second_best_score) >= 0.020
        elif len(scored_candidates) == 1 and is_distant:
            # Single-gallery distant match must be strictly confident (>= base_thresh)
            # to avoid matching un-enrolled passersby when no 2nd candidate exists.
            margin_ok = best_score >= base_thresh

        if best_score >= effective_thresh and margin_ok:
            best_name = best_person["name"]
            is_known = True
            best_is_suspect = best_person.get("is_suspect", False)
            best_threat_level = best_person.get("threat_level", "HIGH" if best_is_suspect else "LOW")
            best_category = best_person.get(
                "category",
                "Wanted / BOLO" if best_is_suspect else "Authorized Personnel",
            )

        is_ambiguous = bool(
            is_known
            and second_best_score > (effective_thresh * 0.85)
            and (best_score - second_best_score) < 0.035
            and best_score < 0.50
        )

        return best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous

    def _detect_faces_in_person_rois(
        self,
        image_bgr: np.ndarray,
        person_boxes: list[Any],
        known_list: list[dict[str, Any]],
        min_match_score: float,
        min_face_size: int = 12,
        det_score_thresh: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Person-Guided Region of Interest (RoI) Face Detection.
        Extracts the head crop directly from native resolution, upscales small crops,
        and runs UniFace detection directly on the focused region.
        """
        if not person_boxes or self.analyzer is None:
            return []

        h, w = image_bgr.shape[:2]
        roi_detections: list[dict[str, Any]] = []

        for pbox in person_boxes:
            try:
                if isinstance(pbox, dict):
                    coords = pbox.get("box") or pbox.get("bbox") or []
                elif isinstance(pbox, (list, tuple, np.ndarray)):
                    coords = pbox
                else:
                    continue

                if len(coords) < 4:
                    continue

                px1, py1, px2, py2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                pw = px2 - px1
                ph = py2 - py1
                if pw < 14 or ph < 20:
                    continue

                roi_x1 = max(0, int(px1 - pw * 0.15))
                roi_y1 = max(0, int(py1 - ph * 0.05))
                roi_x2 = min(w, int(px2 + pw * 0.15))
                roi_y2 = min(h, int(py1 + ph * 0.38))
                roi_w = roi_x2 - roi_x1
                roi_h = roi_y2 - roi_y1

                if roi_w < 12 or roi_h < 12:
                    continue

                head_crop = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
                if head_crop is None or head_crop.size == 0:
                    continue

                scale = 1.0
                if roi_w < 160 or roi_h < 160:
                    scale = min(3.0, max(1.5, 200.0 / max(roi_w, roi_h)))
                    up_w = int(roi_w * scale)
                    up_h = int(roi_h * scale)
                    up_crop = cv2.resize(head_crop, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
                    blurred = cv2.GaussianBlur(up_crop, (0, 0), 1.5)
                    up_crop = cv2.addWeighted(up_crop, 1.35, blurred, -0.35, 0)
                else:
                    up_crop = head_crop
                    up_w, up_h = roi_w, roi_h

                inv_roi_scale = 1.0 / scale if scale > 0 else 1.0

                with self.lock:
                    faces_uf = self.analyzer.analyze(up_crop)

                for face_uf in faces_uf:
                    det_conf = float(face_uf.confidence)
                    if det_conf < det_score_thresh:
                        continue

                    bbox_arr = face_uf.bbox
                    bx1 = roi_x1 + max(0.0, float(bbox_arr[0]) * inv_roi_scale)
                    by1 = roi_y1 + max(0.0, float(bbox_arr[1]) * inv_roi_scale)
                    bx2 = roi_x1 + min(float(roi_w), float(bbox_arr[2]) * inv_roi_scale)
                    by2 = roi_y1 + min(float(roi_h), float(bbox_arr[3]) * inv_roi_scale)
                    fw, fh = bx2 - bx1, by2 - by1

                    if fw < min_face_size or fh < min_face_size:
                        continue

                    feat_512 = None
                    if hasattr(face_uf, "landmarks") and face_uf.landmarks is not None:
                        feat_512 = self.extract_tta_embedding(up_crop, face_uf.landmarks, is_distant=True)
                    if feat_512 is None and face_uf.embedding is not None:
                        feat_512 = face_uf.embedding.astype(np.float32).reshape(1, 512)
                    if feat_512 is None:
                        continue

                    probe_quality = self._score_face_quality(face_uf, image_bgr=up_crop)
                    best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                        self._match_embedding(
                            feat_512, known_list, min_match_score,
                            is_distant=True, face_size=fw,
                            probe_quality=probe_quality,
                        )

                    calibrated_conf = calibrate_arcface_confidence(
                        max(0.0, best_score), threshold=min_match_score
                    )
                    if is_ambiguous:
                        calibrated_conf = min(calibrated_conf, 65.0)

                    best_is_suspect = is_known and bool(
                        next((p.get("is_suspect", False) for p in known_list if p["name"] == best_name), False)
                    )
                    is_threat = bool(is_known and best_is_suspect)
                    class_name = (
                        f"🚨 SUSPECT: {best_name}" if is_threat
                        else (f"Face: {best_name}" if is_known else "Face: Unknown")
                    )

                    roi_detections.append({
                        "name": best_name,
                        "is_known": is_known,
                        "is_ambiguous": is_ambiguous,
                        "confidence": round(det_conf, 4),
                        "match_score": round(max(0.0, best_score), 4),
                        "quality": round(probe_quality, 4),
                        "l2_distance": 0.0,
                        "calibrated_conf": calibrated_conf,
                        "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                        "type": "face",
                        "class_name": class_name,
                        "is_threat": is_threat,
                        "threat_level": best_threat_level if is_threat else "NONE",
                        "category": best_category if is_known else "Unknown",
                        "is_distant": True,
                        "embedding": feat_512.flatten().astype(np.float32),
                    })
            except Exception as roi_err:
                logger.debug("Error processing person RoI for distant face: %s", roi_err)
                continue

        return roi_detections

    def _detect_faces_tiled(
        self,
        image_bgr: np.ndarray,
        known_list: list[dict[str, Any]],
        min_match_score: float,
        min_face_size: int = 12,
        det_score_thresh: float = 0.25,
    ) -> list[dict[str, Any]]:
        """
        Multi-scale tiled face detection for high-resolution cameras when person boxes are unavailable.
        Slices the upper-half surveillance horizons into overlapping tiles.
        """
        if self.analyzer is None:
            return []

        h, w = image_bgr.shape[:2]
        tiled_detections: list[dict[str, Any]] = []

        tiles = [
            (0, 0, int(w * 0.6), int(h * 0.65)),
            (int(w * 0.4), 0, w, int(h * 0.65)),
        ]

        for (tx1, ty1, tx2, ty2) in tiles:
            tile_crop = image_bgr[ty1:ty2, tx1:tx2]
            if tile_crop is None or tile_crop.size == 0:
                continue

            th, tw = tile_crop.shape[:2]
            t_scale = 1.0
            if tw > 960:
                t_scale = 960.0 / tw
                tile_det_img = cv2.resize(tile_crop, (960, int(th * t_scale)))
            else:
                tile_det_img = tile_crop

            t_inv = 1.0 / t_scale if t_scale < 1.0 else 1.0

            try:
                with self.lock:
                    faces_uf = self.analyzer.analyze(tile_det_img)
            except Exception:
                faces_uf = []

            for face_uf in faces_uf:
                if float(face_uf.confidence) < det_score_thresh:
                    continue
                bbox_arr = face_uf.bbox
                bx1 = tx1 + (float(bbox_arr[0]) * t_inv)
                by1 = ty1 + (float(bbox_arr[1]) * t_inv)
                bx2 = tx1 + (float(bbox_arr[2]) * t_inv)
                by2 = ty1 + (float(bbox_arr[3]) * t_inv)
                fw, fh = bx2 - bx1, by2 - by1

                if fw < min_face_size or fh < min_face_size:
                    continue

                feat_512 = None
                if hasattr(face_uf, "landmarks") and face_uf.landmarks is not None:
                    landmarks_native = face_uf.landmarks.copy() * t_inv
                    feat_512 = self.extract_tta_embedding(tile_crop, landmarks_native, is_distant=True)
                if feat_512 is None and face_uf.embedding is not None:
                    feat_512 = face_uf.embedding.astype(np.float32).reshape(1, 512)
                if feat_512 is None:
                    continue

                probe_quality = self._score_face_quality(face_uf, image_bgr=tile_crop)
                best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                    self._match_embedding(
                        feat_512, known_list, min_match_score,
                        is_distant=True, face_size=fw,
                        probe_quality=probe_quality,
                    )

                calibrated_conf = calibrate_arcface_confidence(
                    max(0.0, best_score), threshold=min_match_score
                )
                best_is_suspect = is_known and bool(
                    next((p.get("is_suspect", False) for p in known_list if p["name"] == best_name), False)
                )
                is_threat = bool(is_known and best_is_suspect)
                tiled_detections.append({
                    "name": best_name,
                    "is_known": is_known,
                    "is_ambiguous": is_ambiguous,
                    "confidence": round(float(face_uf.confidence), 4),
                    "match_score": round(max(0.0, best_score), 4),
                    "quality": round(probe_quality, 4),
                    "l2_distance": 0.0,
                    "calibrated_conf": calibrated_conf,
                    "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                    "type": "face",
                    "class_name": f"🚨 SUSPECT: {best_name}" if is_threat else (f"Face: {best_name}" if is_known else "Face: Unknown"),
                    "is_threat": is_threat,
                    "threat_level": best_threat_level if is_threat else "NONE",
                    "category": best_category if is_known else "Unknown",
                    "is_distant": True,
                    "embedding": feat_512.flatten().astype(np.float32),
                })

        return tiled_detections

    def detect_and_recognize(
        self,
        image_bgr: np.ndarray,
        min_match_score: Optional[float] = None,
        min_face_size: int = 18,
        det_score_thresh: float = 0.35,
        use_temporal_smoothing: bool = True,
        max_det_width: int = 960,
        enable_distant_mode: bool = False,
        person_boxes: Optional[list[Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Detect faces in image_bgr and match against enrolled persons using UniFace (512-D).

        Features:
          - UniFace (RetinaFace/SCRFD + AdaFace 512-D embeddings)
          - Person-Guided RoI detection directly on native resolution
          - Multi-scale horizon slicing for distant small faces
          - Test-Time Augmentation (TTA) embedding extraction
          - Adaptive unsharp masking and CLAHE for low-resolution face crops
          - Distance-adaptive match thresholding with separation margin guard
          - Exclusive 1-to-1 identity assignment per frame
          - Temporal identity smoothing to prevent momentary flicker
        """
        if min_match_score is None:
            min_match_score = self.match_threshold

        if enable_distant_mode:
            if min_face_size == 18:
                min_face_size = 12
            if det_score_thresh == 0.35:
                det_score_thresh = 0.25
            if max_det_width == 960:
                max_det_width = 1440

        if image_bgr is None or image_bgr.size == 0 or not self.is_ready or self.analyzer is None:
            return []

        h, w = image_bgr.shape[:2]
        now = time.time()
        raw_detections: list[dict[str, Any]] = []

        with self.lock:
            known_list = list(self.known_persons)

        # ------------------------------------------------------------------ #
        # GLOBAL FULL-FRAME PASS (UniFace)                                   #
        # ------------------------------------------------------------------ #
        scale = 1.0
        if max_det_width > 0 and w > max_det_width:
            scale = max_det_width / float(w)
            det_img = cv2.resize(
                image_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            det_img = image_bgr

        try:
            with self.lock:
                faces_uf = self.analyzer.analyze(det_img)
        except Exception as exc:
            logger.warning("UniFace analyze error: %s", exc)
            if use_temporal_smoothing:
                self._age_tracks()
            return []

        if faces_uf:
            inv_scale = 1.0 / scale if scale < 1.0 else 1.0

            for face_uf in faces_uf:
                det_conf = float(face_uf.confidence)
                if det_conf < det_score_thresh:
                    continue

                bbox_arr = face_uf.bbox
                bx1 = max(0.0, float(bbox_arr[0]) * inv_scale)
                by1 = max(0.0, float(bbox_arr[1]) * inv_scale)
                bx2 = min(float(w), float(bbox_arr[2]) * inv_scale)
                by2 = min(float(h), float(bbox_arr[3]) * inv_scale)
                fw, fh = bx2 - bx1, by2 - by1

                if fw < min_face_size or fh < min_face_size:
                    continue
                aspect = fh / max(1.0, fw)
                if aspect < 0.5 or aspect > 2.0:
                    continue

                is_distant_crop = fw < 60 or fh < 60

                # EDifFIQA probe quality for threshold adaptation
                probe_quality = self._score_face_quality(face_uf, image_bgr=image_bgr)

                feat_512 = None
                if hasattr(face_uf, "landmarks") and face_uf.landmarks is not None:
                    landmarks_native = (face_uf.landmarks * inv_scale).astype(np.float32)
                    feat_512 = self.extract_tta_embedding(image_bgr, landmarks_native, is_distant=is_distant_crop)

                if feat_512 is None and face_uf.embedding is not None:
                    feat_512 = face_uf.embedding.astype(np.float32).reshape(1, 512)

                if feat_512 is None:
                    continue

                best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                    self._match_embedding(
                        feat_512, known_list, min_match_score,
                        is_distant=is_distant_crop, face_size=fw,
                        probe_quality=probe_quality,
                    )

                calibrated_conf = calibrate_arcface_confidence(
                    max(0.0, best_score), threshold=min_match_score
                )
                if is_ambiguous:
                    calibrated_conf = min(calibrated_conf, 65.0)

                best_is_suspect = is_known and bool(
                    next((p.get("is_suspect", False) for p in known_list if p["name"] == best_name), False)
                )
                is_threat = bool(is_known and best_is_suspect)
                class_name = (
                    f"🚨 SUSPECT: {best_name}" if is_threat
                    else (f"Face: {best_name}" if is_known else "Face: Unknown")
                )

                raw_detections.append({
                    "name": best_name,
                    "is_known": is_known,
                    "is_ambiguous": is_ambiguous,
                    "confidence": round(det_conf, 4),
                    "match_score": round(max(0.0, best_score), 4),
                    "quality": round(probe_quality, 4),
                    "l2_distance": 0.0,
                    "calibrated_conf": calibrated_conf,
                    "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                    "type": "face",
                    "class_name": class_name,
                    "is_threat": is_threat,
                    "threat_level": best_threat_level if is_threat else "NONE",
                    "category": best_category if is_known else "Unknown",
                    "is_distant": is_distant_crop,
                    "embedding": feat_512.flatten().astype(np.float32),
                })

        # ------------------------------------------------------------------ #
        # DISTANT ENHANCEMENTS: Person RoIs & Multi-Scale Slicing            #
        # ------------------------------------------------------------------ #
        if person_boxes:
            roi_dets = self._detect_faces_in_person_rois(
                image_bgr=image_bgr,
                person_boxes=person_boxes,
                known_list=known_list,
                min_match_score=min_match_score,
                min_face_size=min_face_size,
                det_score_thresh=det_score_thresh,
            )
            if roi_dets:
                raw_detections.extend(roi_dets)
        elif enable_distant_mode and (w >= 1280 or h >= 720):
            tiled_dets = self._detect_faces_tiled(
                image_bgr=image_bgr,
                known_list=known_list,
                min_match_score=min_match_score,
                min_face_size=min_face_size,
                det_score_thresh=det_score_thresh,
            )
            if tiled_dets:
                raw_detections.extend(tiled_dets)

        if len(raw_detections) > 1:
            raw_detections = deduplicate_face_detections(raw_detections, iou_thresh=0.35)

        if not raw_detections:
            if use_temporal_smoothing:
                self._age_tracks()
            return []

        # Exclusive 1-to-1 Identity Assignment (pre-pass on raw detections):
        raw_detections = self._enforce_one_to_one_identities(raw_detections, min_match_score)

        if not use_temporal_smoothing:
            self._log_events(raw_detections)
            return raw_detections

        smoothed_results = self._apply_temporal_smoothing(raw_detections, now, min_match_score)
        # Final post-smoothing 1-to-1 conflict resolution guarantee
        smoothed_results = self._enforce_one_to_one_identities(smoothed_results, min_match_score)
        self._log_events(smoothed_results)
        return smoothed_results

    def _enforce_one_to_one_identities(
        self,
        detections: list[dict[str, Any]],
        min_match_score: float,
    ) -> list[dict[str, Any]]:
        """
        Enforce physical invariant: One unique identity cannot occupy multiple bodies in the same frame.
        Sorts detections by highest match evidence (match_score, calibrated_conf).
        Any subsequent detection claiming an already-claimed identity is demoted to 'Unknown'.
        """
        if not detections:
            return detections

        claimed_identities: set[str] = set()
        # Sort so the highest-evidence detection claims the identity first
        sorted_indices = sorted(
            range(len(detections)),
            key=lambda i: (
                1 if detections[i].get("is_known") else 0,
                float(detections[i].get("match_score", 0.0)),
                float(detections[i].get("calibrated_conf", 0.0)),
            ),
            reverse=True,
        )

        for idx in sorted_indices:
            det = detections[idx]
            if det.get("is_known"):
                name_key = str(det.get("name", "")).strip().lower()
                if name_key and name_key != "unknown":
                    if name_key not in claimed_identities:
                        claimed_identities.add(name_key)
                    else:
                        # Conflict! Duplicate identity in same frame: demote secondary to Unknown
                        det["is_known"] = False
                        det["is_threat"] = False
                        det["threat_level"] = "NONE"
                        det["name"] = "Unknown"
                        det["class_name"] = "Face: Unknown"
                        det["category"] = "Unknown"
                        det["calibrated_conf"] = calibrate_arcface_confidence(
                            min(det.get("match_score", 0.0), min_match_score * 0.85),
                            threshold=min_match_score,
                        )

        return detections

    def _apply_temporal_smoothing(
        self,
        current_detections: list[dict[str, Any]],
        now: float,
        min_match_score: float,
    ) -> list[dict[str, Any]]:
        self._age_tracks(now)
        matched_track_ids = set()
        final_results = []

        for det in current_detections:
            det_box = det["bbox"]
            best_match_val = 0.0
            best_tid = None

            det_cx = (det_box[0] + det_box[2]) / 2.0
            det_cy = (det_box[1] + det_box[3]) / 2.0
            det_w = max(1.0, det_box[2] - det_box[0])
            det_h = max(1.0, det_box[3] - det_box[1])
            det_diag = max(20.0, np.sqrt(det_w * det_w + det_h * det_h))

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                t_box = track["bbox"]
                iou = compute_iou(det_box, t_box)

                t_cx = (t_box[0] + t_box[2]) / 2.0
                t_cy = (t_box[1] + t_box[3]) / 2.0
                center_dist = np.sqrt((det_cx - t_cx) ** 2 + (det_cy - t_cy) ** 2)

                is_proximate = center_dist <= (det_diag * 1.5)
                if iou >= 0.15 or is_proximate:
                    score = iou if iou > 0.0 else max(0.01, 1.0 - (center_dist / (det_diag * 2.0)))
                    if score > best_match_val:
                        best_match_val = score
                        best_tid = tid

            if best_tid is not None:
                matched_track_ids.add(best_tid)
                track = self.tracks[best_tid]
                track["bbox"] = det_box
                track["last_seen"] = now
                det["track_id"] = best_tid

                track["history"].append({
                    "name": det["name"],
                    "is_known": det["is_known"],
                    "score": det["match_score"],
                    "calibrated": det["calibrated_conf"],
                })
                if len(track["history"]) > 8:
                    track["history"].pop(0)

                known_votes = [h for h in track["history"] if h["is_known"]]
                total_votes = len(track["history"])

                # --- IDENTITY LOCK / HYSTERESIS ---
                # Once a track gets its first confident known detection, we store a
                # "locked_name" (lock fires at identity_lock_threshold=1 known vote).
                # While the lock is active, momentary Unknown frames are overridden
                # with the locked label so the on-screen name never flickers.
                # The lock releases only after `identity_drop_tolerance` consecutive
                # Unknown frames, indicating the person has genuinely left.
                locked_name: Optional[str] = track.get("locked_name")
                locked_score: float = track.get("locked_score", 0.0)
                locked_calibrated: float = track.get("locked_calibrated", 0.0)
                unknown_streak: int = track.get("unknown_streak", 0)

                if det.get("is_known") and len(known_votes) >= self.identity_lock_threshold:
                    # --- LOCK ACQUISITION / REFRESH ---
                    names_so_far = [h["name"] for h in known_votes]
                    majority_name = max(set(names_so_far), key=names_so_far.count)

                    # Only commit the lock when the majority agrees with the current frame's name.
                    # This prevents a late noisy detection from silently stealing the locked identity.
                    if det["name"] == majority_name or names_so_far.count(majority_name) >= self.identity_lock_threshold:
                        track["locked_name"] = majority_name
                        track["locked_score"] = max(h["score"] for h in known_votes if h["name"] == majority_name)
                        track["locked_calibrated"] = max(h["calibrated"] for h in known_votes if h["name"] == majority_name)
                        locked_name = track["locked_name"]
                        locked_score = track["locked_score"]
                        locked_calibrated = track["locked_calibrated"]

                    # Reset unknown streak: this frame is a confirmed known frame
                    track["unknown_streak"] = 0
                    unknown_streak = 0

                elif locked_name is not None:
                    # --- LOCK HOLD: Unknown frame while lock is active ---
                    # Only increment streak once a lock has been established.
                    # (Don't count Unknown frames that arrive before the first confirmed detection.)
                    unknown_streak += 1
                    track["unknown_streak"] = unknown_streak

                    if unknown_streak <= self.identity_drop_tolerance:
                        # Bridge: override Unknown with the locked identity
                        known_match = next(
                            (p for p in self.known_persons if p["name"].lower() == locked_name.lower()), None
                        )
                        if known_match:
                            is_suspect_val = known_match.get("is_suspect", False)
                            threat_lvl_val = known_match.get("threat_level", "HIGH" if is_suspect_val else "LOW")
                            cat_val = known_match.get("category", "Wanted / BOLO" if is_suspect_val else "Authorized Personnel")

                            det["name"] = locked_name
                            det["is_known"] = True
                            det["is_threat"] = is_suspect_val
                            det["threat_level"] = threat_lvl_val if is_suspect_val else "NONE"
                            det["category"] = cat_val
                            det["match_score"] = round(locked_score, 4)
                            det["calibrated_conf"] = round(locked_calibrated, 1)
                            det["class_name"] = f"🚨 SUSPECT: {locked_name}" if is_suspect_val else f"Face: {locked_name}"

                            final_results.append(det)
                            continue  # Skip normal stabilize path for this frame

                    else:
                        # --- LOCK RELEASE: sustained absence, person genuinely left ---
                        track["locked_name"] = None
                        track["locked_score"] = 0.0
                        track["locked_calibrated"] = 0.0
                        track["unknown_streak"] = 0

                else:
                    # --- PRE-LOCK: no lock yet, Unknown frame ---
                    # Don't increment unknown_streak — the lock hasn't been set yet.
                    # If we have any known votes in history, use them as a soft bridge
                    # (handles the alternating hariom/Unknown pattern before lock is set).
                    if known_votes:
                        majority_name = max(
                            set(h["name"] for h in known_votes),
                            key=lambda n: sum(1 for h in known_votes if h["name"] == n)
                        )
                        known_match = next(
                            (p for p in self.known_persons if p["name"].lower() == majority_name.lower()), None
                        )
                        if known_match:
                            is_suspect_val = known_match.get("is_suspect", False)
                            threat_lvl_val = known_match.get("threat_level", "HIGH" if is_suspect_val else "LOW")
                            cat_val = known_match.get("category", "Wanted / BOLO" if is_suspect_val else "Authorized Personnel")
                            avg_score = max(h["score"] for h in known_votes if h["name"] == majority_name)
                            avg_cal = max(h["calibrated"] for h in known_votes if h["name"] == majority_name)

                            det["name"] = majority_name
                            det["is_known"] = True
                            det["is_threat"] = is_suspect_val
                            det["threat_level"] = threat_lvl_val if is_suspect_val else "NONE"
                            det["category"] = cat_val
                            det["match_score"] = round(avg_score, 4)
                            det["calibrated_conf"] = round(avg_cal, 1)
                            det["class_name"] = f"🚨 SUSPECT: {majority_name}" if is_suspect_val else f"Face: {majority_name}"

                            final_results.append(det)
                            continue

                # Normal stabilize path (for current-frame-known detections)
                if det.get("is_known") and known_votes:
                    candidate_votes = list(known_votes)
                    names = [h["name"] for h in candidate_votes]
                    stabilized_name = max(set(names), key=names.count)
                    avg_score = max(h["score"] for h in candidate_votes)
                    avg_calibrated = max(h["calibrated"] for h in candidate_votes)

                    known_match = next((p for p in self.known_persons if p["name"].lower() == stabilized_name.lower()), None)
                    is_suspect_val = known_match.get("is_suspect", False) if known_match else False
                    threat_lvl_val = known_match.get("threat_level", "HIGH" if is_suspect_val else "LOW") if known_match else "NONE"
                    cat_val = known_match.get("category", "Wanted / BOLO" if is_suspect_val else "Authorized Personnel") if known_match else "Unknown"

                    det["name"] = stabilized_name
                    det["is_known"] = True
                    det["is_threat"] = is_suspect_val
                    det["threat_level"] = threat_lvl_val if is_suspect_val else "NONE"
                    det["category"] = cat_val
                    det["match_score"] = round(avg_score, 4)
                    det["calibrated_conf"] = round(avg_calibrated, 1)
                    det["class_name"] = f"🚨 SUSPECT: {stabilized_name}" if is_suspect_val else f"Face: {stabilized_name}"
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                det["track_id"] = tid
                self.tracks[tid] = {
                    "bbox": det_box,
                    "history": [{
                        "name": det["name"],
                        "is_known": det["is_known"],
                        "score": det["match_score"],
                        "calibrated": det["calibrated_conf"],
                    }],
                    "last_seen": now,
                    "locked_name": None,
                    "locked_score": 0.0,
                    "locked_calibrated": 0.0,
                    "unknown_streak": 0,
                }

            final_results.append(det)

        # Enforce 1-to-1 identity exclusivity on final smoothed results
        final_results = self._enforce_one_to_one_identities(final_results, min_match_score)

        # Sync track history if any track's identity was revoked during conflict resolution
        for det in final_results:
            if not det.get("is_known"):
                tid = det.get("track_id")
                if tid is not None and tid in self.tracks and self.tracks[tid]["history"]:
                    last_h = self.tracks[tid]["history"][-1]
                    if last_h.get("is_known") and last_h.get("name") != "Unknown":
                        last_h["is_known"] = False
                        last_h["name"] = "Unknown"
                    # Also release lock if 1-to-1 enforcement revoked this identity
                    if self.tracks[tid].get("locked_name") is not None:
                        self.tracks[tid]["locked_name"] = None
                        self.tracks[tid]["locked_score"] = 0.0
                        self.tracks[tid]["locked_calibrated"] = 0.0

        return final_results

    def _age_tracks(self, now: Optional[float] = None) -> None:
        """Evict stale tracks that haven't been detected for more than 4 seconds.

        4s gives enough buffer to survive brief occlusions (person blocked by car/post)
        without destroying the track history and identity lock that prevents flickering.
        """
        if now is None:
            now = time.time()
        stale_tids = [tid for tid, track in self.tracks.items() if (now - track["last_seen"]) > 4.0]
        for tid in stale_tids:
            del self.tracks[tid]

    def _log_events(self, detections: list[dict[str, Any]]) -> None:
        """Record detection events into the live circular operational buffer."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for d in detections:
            event = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": now_str,
                "name": d["name"],
                "is_known": d["is_known"],
                "match_score": d["match_score"],
                "confidence": d["confidence"],
                "calibrated_conf": d["calibrated_conf"],
                "bbox": d["bbox"],
                "is_threat": d["is_threat"],
                "threat_level": d["threat_level"],
                "category": d["category"],
            }
            self.recent_events.append(event)
            if len(self.recent_events) > self.max_events:
                self.recent_events.pop(0)

    def get_recent_events(self) -> list[dict[str, Any]]:
        """Return the recent face detections log in reverse chronological order."""
        return list(reversed(self.recent_events))

    def draw_faces(
        self,
        image_bgr: np.ndarray,
        faces: list[dict[str, Any]],
        show_metrics: bool = True,
    ) -> np.ndarray:
        """Annotate face bounding boxes, identities, and threat classifications."""
        annotated = image_bgr.copy()
        for f in faces:
            bbox = f.get("bbox", [])
            if len(bbox) < 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            name = f.get("name", "Unknown")
            is_breach = bool(f.get("is_breach", False))
            breach_lbl = str(f.get("breach_label", "PERIMETER"))
            is_threat = bool(f.get("is_threat", False)) or is_breach
            is_known = bool(f.get("is_known", False))
            conf_pct = float(f.get("calibrated_conf", 0.0))

            if is_breach:
                box_color = (0, 0, 255)       # Bright Warning Red
                tag_bg = (0, 0, 200)
            elif is_threat:
                box_color = (0, 0, 220)       # Crimson Red
                tag_bg = (0, 0, 180)
            elif is_known:
                box_color = (60, 220, 60)     # Emerald Green
                tag_bg = (40, 160, 40)
            else:
                box_color = (180, 180, 180)   # Neutral Gray
                tag_bg = (100, 100, 100)

            box_thickness = 3 if is_breach else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thickness)

            if is_breach:
                label = f"🚨 BREACH: {breach_lbl} [{name}]"
            elif show_metrics:
                label = f"{name} {int(conf_pct)}%"
            else:
                label = name
            (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

            ty2 = max(y1, lh + 6)
            ty1 = ty2 - lh - 6
            cv2.rectangle(annotated, (x1, ty1), (x1 + lw + 8, ty2), tag_bg, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, ty2 - base - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated


face_service = FaceService()
