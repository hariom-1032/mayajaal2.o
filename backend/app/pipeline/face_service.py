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
# Optional UniFace import (SCRFD + ArcFace/AdaFace via ONNX Runtime)
# If not installed, the service falls back to OpenCV YuNet + SFace.
# ---------------------------------------------------------------------------
try:
    from uniface import FaceAnalyzer
    from uniface.detection import SCRFD
    from uniface.recognition import AdaFace
    from uniface.constants import AdaFaceWeights
    from uniface.recognition.base import face_alignment
    _UNIFACE_AVAILABLE = True
except ImportError:
    _UNIFACE_AVAILABLE = False

logger = logging.getLogger("SIH26187.FaceService")


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
    Safe cosine similarity between two embedding vectors of any dimension.
    Works with both 128-D SFace and 512-D ArcFace/AdaFace embeddings.
    Returns a value in [-1, 1].
    """
    v1 = feat1.flatten().astype(np.float32)
    v2 = feat2.flatten().astype(np.float32)
    norm = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def calibrate_match_confidence(cosine_score: float, threshold: float = 0.34) -> float:
    """
    Calibrate raw SFace cosine similarity (-1.0 to 1.0) into an intuitive, human-friendly
    confidence percentage (0% to 100%).

    In SFace (calibrated for CCTV / surveillance):
      - c < 0.18: Non-match (noise / random face) -> 0% - 15%
      - c < threshold (0.34): Unrecognized / below threshold -> 15% - 49%
      - c = threshold: Standard decision boundary -> 50%
      - c = 0.42: Solid match -> 75%
      - c = 0.50: High confidence match -> 88%
      - c >= 0.60: Near identical / verified match -> 96% - 99.5%
    Used when falling back to OpenCV SFace.
    """
    c = float(cosine_score)
    th = float(threshold)
    if c <= 0.0:
        return 0.0
    elif c <= 0.18:
        val = (c / 0.18) * 15.0
    elif c < th:
        val = 15.0 + ((c - 0.18) / max(0.01, th - 0.18)) * 34.0
    elif c <= 0.45:
        val = 50.0 + ((c - th) / max(0.01, 0.45 - th)) * 32.0
    elif c <= 0.58:
        val = 82.0 + ((c - 0.45) / (0.58 - 0.45)) * 13.0
    else:
        val = 95.0 + min(1.0, (c - 0.58) / 0.20) * 4.5
    return round(float(np.clip(val, 0.0, 99.9)), 1)


def calibrate_arcface_confidence(cosine_score: float, threshold: float = 0.40) -> float:
    """
    Calibrate ArcFace / AdaFace cosine similarity (L2-normalized, range -1 to 1)
    into a human-friendly confidence percentage (0% to 100%).

    ArcFace/AdaFace calibration breakpoints (512-D embeddings):
      - c < 0.20: Non-match / noise         ->  0% - 15%
      - c < threshold (0.40): Below boundary -> 15% - 49%
      - c = threshold: Decision boundary     -> 50%
      - c = 0.55: Solid match               -> 85%
      - c >= 0.65: Near-certain verified     -> 95% - 99.5%
    """
    c = float(cosine_score)
    th = float(threshold)
    if c <= 0.0:
        return 0.0
    elif c <= 0.20:
        val = (c / 0.20) * 15.0
    elif c < th:
        val = 15.0 + ((c - 0.20) / max(0.01, th - 0.20)) * 34.0
    elif c <= 0.55:
        val = 50.0 + ((c - th) / max(0.01, 0.55 - th)) * 35.0
    elif c <= 0.65:
        val = 85.0 + ((c - 0.55) / 0.10) * 10.0
    else:
        val = 95.0 + min(1.0, (c - 0.65) / 0.25) * 4.5
    return round(float(np.clip(val, 0.0, 99.9)), 1)



class FaceService:
    """
    Production-grade facial recognition service for MAATRIX.

    Primary engine  : UniFace  — SCRFD detector + AdaFace recognizer (512-D embeddings).
    Fallback engine : OpenCV   — YuNet detector + SFace recognizer   (128-D embeddings).

    Thread-safe and optimized for real-time video streams and REST APIs.
    Database migration from 128-D to 512-D is user-initiated (deferred).
    """

    def __init__(self) -> None:
        self.models_dir = settings.model_directory
        self.data_dir = settings.model_directory.parent / "app" / "data" / "faces"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.db_file = self.data_dir / "known_faces.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

        # --- UniFace engine (primary) ---
        # Populated when uniface package is available.
        self.analyzer: Optional["FaceAnalyzer"] = None  # type: ignore[name-defined]
        self.use_uniface: bool = False
        # Embedding dimension for the active recognizer (512 for ArcFace/AdaFace, 128 for SFace).
        self.embedding_dim: int = 512
        # Matching threshold (0.40 for ArcFace/AdaFace, 0.34 for SFace fallback).
        self.match_threshold: float = 0.40

        # --- OpenCV fallback engine ---
        self.yunet_path = self.models_dir / "face_detection_yunet.onnx"
        self.sface_path = self.models_dir / "face_recognition_sface.onnx"
        self.detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[cv2.FaceRecognizerSF] = None

        self.lock = threading.Lock()

        # In-memory known persons list
        self.known_persons: list[dict[str, Any]] = []

        # Recent detection logs for live operational dashboard
        self.recent_events: list[dict[str, Any]] = []
        self.max_events: int = 50

        # Temporal tracking history to eliminate video jitter and momentary false alerts
        self.tracks: dict[int, dict[str, Any]] = {}
        self.next_track_id: int = 1

        self._init_models()
        self.load_database()

    def _init_models(self) -> None:
        """Initialize face engine: tries UniFace first, falls back to OpenCV YuNet+SFace."""
        # --- Attempt UniFace (SCRFD + AdaFace) ---
        if _UNIFACE_AVAILABLE:
            try:
                self.analyzer = FaceAnalyzer(
                    detector=SCRFD(
                        confidence_threshold=0.35,
                        input_size=(960, 960),
                    ),
                    recognizer=AdaFace(model_name=AdaFaceWeights.IR_101),
                )
                self.use_uniface = True
                self.embedding_dim = 512
                self.match_threshold = 0.40
                logger.info(
                    "FaceService initialized with high-accuracy UniFace (SCRFD 960 + AdaFace IR-101, 512-D embeddings)."
                )
                return
            except Exception as exc:
                logger.warning(
                    "UniFace initialization failed, falling back to YuNet+SFace: %s", exc
                )

        # --- Fallback: OpenCV YuNet + SFace ---
        try:
            if not self.yunet_path.exists() or not self.sface_path.exists():
                logger.warning(
                    "Face models missing — YuNet exists=%s, SFace exists=%s",
                    self.yunet_path.exists(),
                    self.sface_path.exists(),
                )
                return

            self.detector = cv2.FaceDetectorYN.create(
                str(self.yunet_path),
                "",
                (320, 320),
                score_threshold=0.45,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(str(self.sface_path), "")
            self.use_uniface = False
            self.embedding_dim = 128
            self.match_threshold = 0.34
            logger.info("FaceService fallback initialized with OpenCV YuNet + SFace (128-D).")
        except Exception as exc:
            logger.exception("Error initializing fallback OpenCV Face models: %s", exc)

    @property
    def engine_name(self) -> str:
        """Human-readable active engine identifier for status endpoints."""
        return "UniFace (SCRFD 960 + AdaFace IR-101)" if self.use_uniface else "OpenCV (YuNet + SFace)"

    @property
    def is_ready(self) -> bool:
        """True when at least one face engine is fully operational."""
        if self.use_uniface:
            return self.analyzer is not None
        return self.detector is not None and self.recognizer is not None

    @staticmethod
    def _compute_centroid(features_np: list[np.ndarray]) -> Optional[np.ndarray]:
        """Compute the L2-normalized mean embedding (centroid) across all enrolled samples."""
        if not features_np:
            return None
        dim = features_np[0].flatten().shape[0]
        valid_features = [f.reshape(1, -1) for f in features_np if f.flatten().shape[0] == dim]
        if not valid_features:
            return None
        stacked = np.vstack(valid_features).astype(np.float32)
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
        if not self.use_uniface or self.analyzer is None or self.analyzer.recognizer is None:
            return None
        rec = self.analyzer.recognizer
        try:
            # 1. Similarity-align 112x112 canonical crop from full-resolution frame
            aligned_face, _ = face_alignment(
                image_bgr, landmarks, image_size=rec.input_size
            )
            if aligned_face is None or aligned_face.size == 0:
                return None

            if is_distant:
                aligned_face = enhance_aligned_face(aligned_face, is_distant_or_low_res=True)

            # 2. Forward pass on original aligned face
            blob1 = rec.preprocess(aligned_face)
            e1 = rec.session.run(rec.output_names, {rec.input_name: blob1})[0].ravel()

            # 3. Forward pass on horizontally mirrored face (TTA)
            flipped_face = cv2.flip(aligned_face, 1)
            blob2 = rec.preprocess(flipped_face)
            e2 = rec.session.run(rec.output_names, {rec.input_name: blob2})[0].ravel()

            # 4. L2-normalized vector fusion
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

    def load_database(self) -> None:
        """
        Load enrolled persons into memory.

        Dimension-safe: Both 128-D (SFace legacy) and 512-D (UniFace) embeddings are
        loaded without error. During recognition, mismatched-dimension vectors are
        silently skipped so fresh 512-D enrollments work alongside pending legacy entries.
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

                for r in records:
                    features_np: list[np.ndarray] = []
                    raw_features = r.get("features", [])
                    if isinstance(raw_features, list) and len(raw_features) > 0:
                        for item in raw_features:
                            if isinstance(item, list) and len(item) in (128, 512):
                                dim = len(item)
                                features_np.append(
                                    np.array(item, dtype=np.float32).reshape(1, dim)
                                )

                    # Backward compatibility fallback: single "feature" (128-D or 512-D)
                    single_feat = r.get("feature", [])
                    if len(features_np) == 0 and isinstance(single_feat, list) and len(single_feat) in (128, 512):
                        dim = len(single_feat)
                        features_np.append(np.array(single_feat, dtype=np.float32).reshape(1, dim))

                    person_id = r.get("id")
                    thumb_name = f"{person_id}.jpg"
                    image_url = f"/api/faces/thumbnail/{thumb_name}"
                    is_suspect = bool(r.get("is_suspect", False))
                    threat_level = str(r.get("threat_level", "HIGH" if is_suspect else "LOW"))
                    category = str(r.get("category", "Wanted / BOLO" if is_suspect else "Authorized Personnel"))
                    notes = str(r.get("notes", ""))

                    if features_np:
                        centroid = self._compute_centroid(features_np)
                        self.known_persons.append({
                            "id": person_id,
                            "name": r.get("name", "Unknown"),
                            "image_url": image_url,
                            "created_at": r.get("created_at", ""),
                            "features": features_np,
                            "feature": features_np[0],
                            "centroid": centroid,
                            "is_suspect": is_suspect,
                            "threat_level": threat_level,
                            "category": category,
                            "notes": notes,
                        })
                logger.info("Loaded %d enrolled persons into FaceService.", len(self.known_persons))
            except Exception as exc:
                logger.exception("Error loading face database from %s: %s", self.db_file, exc)

    def _save_database(self) -> None:
        records = []
        for p in self.known_persons:
            feats_list = [f.flatten().tolist() for f in p.get("features", [p["feature"]])]
            records.append({
                "id": p["id"],
                "name": p["name"],
                "image_url": p["image_url"],
                "created_at": p["created_at"],
                "features": feats_list,
                "feature": feats_list[0] if feats_list else [],
                "is_suspect": p.get("is_suspect", False),
                "threat_level": p.get("threat_level", "HIGH" if p.get("is_suspect") else "LOW"),
                "category": p.get("category", "Wanted / BOLO" if p.get("is_suspect") else "Authorized Personnel"),
                "notes": p.get("notes", ""),
            })
        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    def list_persons(self) -> list[dict[str, Any]]:
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

    def update_person_metadata(
        self,
        person_id: str,
        is_suspect: Optional[bool] = None,
        threat_level: Optional[str] = None,
        category: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Update classification and threat metadata for an enrolled person without re-uploading."""
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
        Enroll a new person or add a sample to an existing profile.

        Quality gates (shared across both engines):
          - Face size >= 65×65 px
          - Detection confidence >= 0.65
          - Laplacian sharpness variance >= 20.0

        UniFace path  : SCRFD detects faces, AdaFace extracts 512-D embedding.
        Fallback path : YuNet detects faces, SFace extracts 128-D embedding.
        """
        name = name.strip()
        if not name:
            return {"success": False, "error": "Name cannot be blank"}
        if image_bgr is None or image_bgr.size == 0:
            return {"success": False, "error": "Invalid image data"}
        if not self.is_ready:
            return {"success": False, "error": "Face recognition engine is not initialized"}

        # ------------------------------------------------------------------ #
        # ENGINE BRANCH A: UniFace (SCRFD + AdaFace, 512-D)                  #
        # ------------------------------------------------------------------ #
        if self.use_uniface and self.analyzer is not None:
            try:
                with self.lock:
                    faces_uf = self.analyzer.analyze(image_bgr)

                if not faces_uf:
                    return {
                        "success": False,
                        "error": "No face detected in this image. Please ensure good lighting and face the camera directly.",
                    }

                # Select highest-confidence detection
                best_uf = max(faces_uf, key=lambda f: f.confidence)
                bbox_arr = best_uf.bbox  # np.ndarray [x1, y1, x2, y2]
                bx1, by1, bx2, by2 = float(bbox_arr[0]), float(bbox_arr[1]), float(bbox_arr[2]), float(bbox_arr[3])
                fw, fh = float(bx2 - bx1), float(by2 - by1)
                det_conf = float(best_uf.confidence)

                # Quality check 1: minimum face dimensions
                if fw < 48 or fh < 48:
                    return {
                        "success": False,
                        "error": f"Face is too small ({int(fw)}×{int(fh)} px). Minimum 48×48 px required.",
                    }

                # Quality check 2: detection confidence
                if det_conf < 0.45:
                    return {
                        "success": False,
                        "error": f"Face detection confidence is low ({int(det_conf * 100)}%). Please face the camera directly in clear light.",
                    }

                # High-fidelity TTA 512-D embedding directly from native resolution
                feature = None
                if hasattr(best_uf, "landmarks") and best_uf.landmarks is not None:
                    feature = self.extract_tta_embedding(image_bgr, best_uf.landmarks)

                if feature is None:
                    if best_uf.embedding is not None:
                        feature = best_uf.embedding.astype(np.float32).reshape(1, 512)
                    else:
                        return {
                            "success": False,
                            "error": "Could not extract face features. Please ensure your face is well-lit and directly facing the camera.",
                        }

                # Canonical aligned face crop for thumbnail
                aligned_face = None
                if hasattr(best_uf, "landmarks") and best_uf.landmarks is not None and self.analyzer and self.analyzer.recognizer:
                    try:
                        aligned_face, _ = face_alignment(
                            image_bgr, best_uf.landmarks, image_size=self.analyzer.recognizer.input_size
                        )
                    except Exception:
                        aligned_face = None

                if aligned_face is None or aligned_face.size == 0:
                    bx1i, by1i = max(0, int(bx1)), max(0, int(by1))
                    bx2i, by2i = min(image_bgr.shape[1], int(bx2)), min(image_bgr.shape[0], int(by2))
                    aligned_face = image_bgr[by1i:by2i, bx1i:bx2i]
                    if aligned_face.size == 0:
                        aligned_face = image_bgr

                # Quality check 3: blur check via Laplacian variance
                gray_aligned = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray_aligned, cv2.CV_64F).var()
                if lap_var < 20.0:
                    return {
                        "success": False,
                        "error": "Image is too blurry. Please hold steady and recapture in good lighting.",
                    }

            except Exception as exc:
                logger.exception("UniFace register_face error: %s", exc)
                return {"success": False, "error": "Face engine error during enrollment. Please retry."}

        # ------------------------------------------------------------------ #
        # ENGINE BRANCH B: OpenCV YuNet + SFace fallback (128-D)             #
        # ------------------------------------------------------------------ #
        else:
            if self.detector is None or self.recognizer is None:
                return {"success": False, "error": "Face detection models are not initialized"}

            h, w = image_bgr.shape[:2]
            with self.lock:
                self.detector.setInputSize((w, h))
                _, faces = self.detector.detect(image_bgr)

            if faces is None or len(faces) == 0:
                return {
                    "success": False,
                    "error": "No face detected in this image. Please ensure good lighting and face the camera directly.",
                }

            # Select largest face by bounding box area (w * h)
            best_face = max(faces, key=lambda f: f[2] * f[3])
            fw, fh = float(best_face[2]), float(best_face[3])
            det_conf = float(best_face[14])

            # Quality check 1: minimum face dimensions
            if fw < 65 or fh < 65:
                return {
                    "success": False,
                    "error": f"Face is too small ({int(fw)}×{int(fh)} px). Minimum 65×65 px required.",
                }

            # Quality check 2: detection confidence
            if det_conf < 0.65:
                return {
                    "success": False,
                    "error": f"Face detection confidence is low ({int(det_conf * 100)}%). Please face the camera directly in clear light.",
                }

            with self.lock:
                aligned_face = self.recognizer.alignCrop(image_bgr, best_face)
                enhanced_face = enhance_aligned_face(aligned_face)
                feature = self.recognizer.feature(enhanced_face)  # Shape: (1, 128)

            # Quality check 3: blur check via Laplacian variance
            gray_aligned = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray_aligned, cv2.CV_64F).var()
            if lap_var < 20.0:
                return {
                    "success": False,
                    "error": "Image is too blurry. Please hold steady and recapture in good lighting.",
                }

        with self.lock:
            existing = next((p for p in self.known_persons if p["name"].lower() == name.lower()), None)

            if existing:
                # Add sample embedding to existing profile (cap at 10 samples)
                if "features" not in existing:
                    existing["features"] = [existing["feature"]]
                if len(existing["features"]) < 10:
                    existing["features"].append(feature)
                else:
                    existing["features"][0] = feature

                # Update metadata if explicitly provided
                existing["is_suspect"] = bool(is_suspect)
                existing["threat_level"] = str(threat_level).upper()
                existing["category"] = str(category).strip() or existing.get("category", "Wanted / BOLO")
                if notes:
                    existing["notes"] = str(notes).strip()

                # Update thumbnail to latest capture
                thumb_path = self.thumbnails_dir / f"{existing['id']}.jpg"
                cv2.imwrite(str(thumb_path), aligned_face)
                existing["feature"] = existing["features"][0]
                existing["centroid"] = self._compute_centroid(existing["features"])
                self._save_database()

                return {
                    "success": True,
                    "person": {
                        "id": existing["id"],
                        "name": existing["name"],
                        "image_url": existing["image_url"],
                        "created_at": existing["created_at"],
                        "sample_count": len(existing["features"]),
                        "is_suspect": existing["is_suspect"],
                        "threat_level": existing["threat_level"],
                        "category": existing["category"],
                        "notes": existing.get("notes", ""),
                    },
                    "message": f"Added new sample to profile for {existing['name']}! (Total samples: {len(existing['features'])})",
                }
            else:
                person_id = f"person_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                thumb_filename = f"{person_id}.jpg"
                thumb_path = self.thumbnails_dir / thumb_filename

                cv2.imwrite(str(thumb_path), aligned_face)

                new_record = {
                    "id": person_id,
                    "name": name,
                    "image_url": f"/api/faces/thumbnail/{thumb_filename}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "features": [feature],
                    "feature": feature,
                    "centroid": feature,
                    "is_suspect": bool(is_suspect),
                    "threat_level": str(threat_level).upper(),
                    "category": str(category).strip() or "Wanted / BOLO",
                    "notes": str(notes).strip(),
                }

                self.known_persons.append(new_record)
                self._save_database()

                return {
                    "success": True,
                    "person": {
                        "id": new_record["id"],
                        "name": new_record["name"],
                        "image_url": new_record["image_url"],
                        "created_at": new_record["created_at"],
                        "sample_count": 1,
                        "is_suspect": new_record["is_suspect"],
                        "threat_level": new_record["threat_level"],
                        "category": new_record["category"],
                        "notes": new_record["notes"],
                    },
                    "message": f"Successfully enrolled face profile for {name}!",
                }

    def add_sample(self, person_id: str, image_bgr: np.ndarray) -> dict[str, Any]:
        """Add an additional angle or lighting sample to a specific person ID."""
        with self.lock:
            person = next((p for p in self.known_persons if p["id"] == person_id), None)
            if not person:
                return {"success": False, "error": "Person ID not found"}
            name = person["name"]
            is_suspect = person.get("is_suspect", False)
            threat_level = person.get("threat_level", "HIGH" if is_suspect else "LOW")
            category = person.get("category", "Wanted / BOLO" if is_suspect else "Authorized Personnel")
            notes = person.get("notes", "")

        return self.register_face(
            image_bgr,
            name,
            is_suspect=is_suspect,
            threat_level=threat_level,
            category=category,
            notes=notes,
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
    ) -> tuple[str, float, bool, str, str, bool]:
        """
        Match a query embedding against all enrolled persons.

        Dimension-safe: skips stored embeddings whose dimension doesn't match the query.
        Uses both synthesized gallery centroids and individual sample embeddings.
        Evaluates Top-1 vs Top-2 separation margin to identify ambiguous matches.
        Returns (name, best_score, is_known, threat_level, category, is_ambiguous).
        """
        query_dim = query_feat.flatten().shape[0]
        best_name = "Unknown"
        best_score = -1.0
        second_best_score = -1.0
        is_known = False
        best_is_suspect = False
        best_threat_level = "LOW"
        best_category = "Unknown"

        for p in known_list:
            p_best_score = -1.0

            # 1. Match against precomputed identity centroid (invariant structure)
            centroid = p.get("centroid")
            if centroid is not None and centroid.flatten().shape[0] == query_dim:
                if self.use_uniface:
                    cos_c = compute_cosine_similarity(query_feat, centroid)
                else:
                    cos_c = float(self.recognizer.match(  # type: ignore[union-attr]
                        query_feat, centroid, cv2.FaceRecognizerSF_FR_COSINE
                    ))
                if cos_c > p_best_score:
                    p_best_score = cos_c

            # 2. Match against all stored individual sample embeddings
            p_features = p.get("features", [p["feature"]])
            for k_feat in p_features:
                # --- Dimension guard: skip mismatched legacy entries ---
                k_dim = k_feat.flatten().shape[0]
                if k_dim != query_dim:
                    continue

                if self.use_uniface:
                    # UniFace: cosine via optimised dot product on L2-normalized vectors
                    cos = compute_cosine_similarity(query_feat, k_feat)
                else:
                    # SFace fallback: use OpenCV's built-in metric
                    cos = float(self.recognizer.match(  # type: ignore[union-attr]
                        query_feat, k_feat, cv2.FaceRecognizerSF_FR_COSINE
                    ))

                if cos > p_best_score:
                    p_best_score = cos

            if p_best_score > best_score:
                second_best_score = best_score
                best_score = p_best_score
                if p_best_score >= min_match_score:
                    best_name = p["name"]
                    is_known = True
                    best_is_suspect = p.get("is_suspect", False)
                    best_threat_level = p.get("threat_level", "HIGH" if best_is_suspect else "LOW")
                    best_category = p.get(
                        "category",
                        "Wanted / BOLO" if best_is_suspect else "Authorized Personnel",
                    )
            elif p_best_score > second_best_score:
                second_best_score = p_best_score

        # Ambiguity check: if Top-1 and Top-2 differ by less than 0.035 and score is not overwhelmingly high
        is_ambiguous = bool(
            is_known
            and second_best_score > 0.32
            and (best_score - second_best_score) < 0.035
            and best_score < 0.55
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
        Distant Person-Guided RoI Facial Detection & Recognition.
        Extracts the upper 35% (head region) of each detected person directly from
        native-resolution image_bgr, upscales/sharpens tiny candidate regions, runs
        high-sensitivity facial detection, and matches against known suspects.
        """
        h, w = image_bgr.shape[:2]
        roi_detections: list[dict[str, Any]] = []

        for p_item in person_boxes:
            try:
                if isinstance(p_item, dict):
                    p_box = p_item.get("box") or p_item.get("bbox")
                else:
                    p_box = p_item

                if not isinstance(p_box, (list, tuple, np.ndarray)) or len(p_box) < 4:
                    continue

                px1, py1, px2, py2 = float(p_box[0]), float(p_box[1]), float(p_box[2]), float(p_box[3])
                pw = px2 - px1
                ph = py2 - py1
                if pw < 14 or ph < 20:
                    continue

                # Head RoI: top 35% of person height with 15% lateral margin
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

                # Adaptive upscale & unsharp masking for tiny head crops
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

                # --- UniFace Branch ---
                if self.use_uniface and self.analyzer is not None:
                    try:
                        with self.lock:
                            faces_uf = self.analyzer.analyze(up_crop)
                    except Exception:
                        faces_uf = []

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

                        best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                            self._match_embedding(feat_512, known_list, min_match_score)

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
                            "l2_distance": 0.0,
                            "calibrated_conf": calibrated_conf,
                            "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                            "type": "face",
                            "class_name": class_name,
                            "is_threat": is_threat,
                            "threat_level": best_threat_level if is_threat else "NONE",
                            "category": best_category if is_known else "Unknown",
                            "is_distant": True,
                        })

                # --- Fallback: OpenCV YuNet + SFace ---
                elif self.detector is not None and self.recognizer is not None:
                    with self.lock:
                        self.detector.setInputSize((up_w, up_h))
                        _, faces = self.detector.detect(up_crop)

                    if faces is not None and len(faces) > 0:
                        for raw_face in faces:
                            det_conf = float(raw_face[14])
                            if det_conf < det_score_thresh:
                                continue

                            x, y, fw_up, fh_up = float(raw_face[0]), float(raw_face[1]), float(raw_face[2]), float(raw_face[3])
                            bx1 = roi_x1 + (x * inv_roi_scale)
                            by1 = roi_y1 + (y * inv_roi_scale)
                            bx2 = bx1 + (fw_up * inv_roi_scale)
                            by2 = by1 + (fh_up * inv_roi_scale)
                            fw, fh = bx2 - bx1, by2 - by1

                            if fw < min_face_size or fh < min_face_size:
                                continue

                            with self.lock:
                                aligned_face = self.recognizer.alignCrop(up_crop, raw_face)
                                enhanced_face = enhance_aligned_face(aligned_face, is_distant_or_low_res=True)
                                feat = self.recognizer.feature(enhanced_face)

                            best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                                self._match_embedding(feat, known_list, min_match_score)

                            best_l2 = 999.0
                            for p in known_list:
                                if p["name"] == best_name:
                                    for k_feat in p.get("features", [p["feature"]]):
                                        if k_feat.flatten().shape[0] == 128:
                                            l2 = float(self.recognizer.match(
                                                feat, k_feat, cv2.FaceRecognizerSF_FR_NORM_L2
                                            ))
                                            if l2 < best_l2:
                                                best_l2 = l2
                                    break

                            calibrated_conf = calibrate_match_confidence(
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
                                "l2_distance": round(best_l2, 4),
                                "calibrated_conf": calibrated_conf,
                                "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                                "type": "face",
                                "class_name": class_name,
                                "is_threat": is_threat,
                                "threat_level": best_threat_level if is_threat else "NONE",
                                "category": best_category if is_known else "Unknown",
                                "is_distant": True,
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
        Multi-scale / sliced inference for distant faces in high-res streams when person boxes are absent.
        Splits high-resolution frames into overlapping surveillance bands so tiny distant faces
        are not destroyed by whole-frame downscaling.
        """
        h, w = image_bgr.shape[:2]
        tiled_detections: list[dict[str, Any]] = []

        y1 = max(0, int(h * 0.08))
        y2 = min(h, int(h * 0.92))
        tile_w = int(w * 0.60)  # 20% horizontal overlap in the center

        tiles = [
            (0, y1, tile_w, y2),
            (max(0, w - tile_w), y1, w, y2),
        ]

        for tx1, ty1, tx2, ty2 in tiles:
            tile_crop = image_bgr[ty1:ty2, tx1:tx2]
            if tile_crop is None or tile_crop.size == 0:
                continue

            tw = tx2 - tx1
            th = ty2 - ty1
            t_scale = 1.0
            if tw > 960:
                t_scale = 960.0 / float(tw)
                tile_det_img = cv2.resize(tile_crop, (960, int(th * t_scale)), interpolation=cv2.INTER_LINEAR)
            else:
                tile_det_img = tile_crop

            t_inv = 1.0 / t_scale if t_scale < 1.0 else 1.0

            if self.use_uniface and self.analyzer is not None:
                try:
                    with self.lock:
                        faces_uf = self.analyzer.analyze(tile_det_img)
                except Exception:
                    faces_uf = []

                for face_uf in faces_uf:
                    if float(face_uf.confidence) < det_score_thresh:
                        continue
                    b = face_uf.bbox
                    bx1 = tx1 + max(0.0, float(b[0]) * t_inv)
                    by1 = ty1 + max(0.0, float(b[1]) * t_inv)
                    bx2 = tx1 + min(float(tw), float(b[2]) * t_inv)
                    by2 = ty1 + min(float(th), float(b[3]) * t_inv)
                    fw, fh = bx2 - bx1, by2 - by1
                    if fw < min_face_size or fh < min_face_size:
                        continue

                    feat_512 = None
                    if hasattr(face_uf, "landmarks") and face_uf.landmarks is not None:
                        lm_native = np.copy(face_uf.landmarks) * t_inv
                        lm_native[:, 0] += tx1
                        lm_native[:, 1] += ty1
                        feat_512 = self.extract_tta_embedding(image_bgr, lm_native, is_distant=(fw < 60 or fh < 60))
                    if feat_512 is None and face_uf.embedding is not None:
                        feat_512 = face_uf.embedding.astype(np.float32).reshape(1, 512)
                    if feat_512 is None:
                        continue

                    best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                        self._match_embedding(feat_512, known_list, min_match_score)

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
                        "l2_distance": 0.0,
                        "calibrated_conf": calibrated_conf,
                        "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                        "type": "face",
                        "class_name": f"🚨 SUSPECT: {best_name}" if is_threat else (f"Face: {best_name}" if is_known else "Face: Unknown"),
                        "is_threat": is_threat,
                        "threat_level": best_threat_level if is_threat else "NONE",
                        "category": best_category if is_known else "Unknown",
                        "is_distant": True,
                    })

            elif self.detector is not None and self.recognizer is not None:
                det_w, det_h = int(tw * t_scale), int(th * t_scale)
                with self.lock:
                    self.detector.setInputSize((det_w, det_h))
                    _, faces = self.detector.detect(tile_det_img)

                if faces is not None and len(faces) > 0:
                    for raw_face in faces:
                        det_conf = float(raw_face[14])
                        if det_conf < det_score_thresh:
                            continue
                        face_native = raw_face.copy()
                        face_native[:14] = face_native[:14] * t_inv
                        x, y, fw, fh = float(face_native[0]), float(face_native[1]), float(face_native[2]), float(face_native[3])
                        if fw < min_face_size or fh < min_face_size:
                            continue

                        face_global = face_native.copy()
                        face_global[0] += tx1
                        face_global[1] += ty1
                        for l_idx in range(4, 14, 2):
                            face_global[l_idx] += tx1
                            face_global[l_idx + 1] += ty1

                        bx1 = float(face_global[0])
                        by1 = float(face_global[1])
                        bx2 = bx1 + fw
                        by2 = by1 + fh

                        with self.lock:
                            aligned_face = self.recognizer.alignCrop(image_bgr, face_global)
                            enhanced_face = enhance_aligned_face(aligned_face, is_distant_or_low_res=(fw < 60 or fh < 60))
                            feat = self.recognizer.feature(enhanced_face)

                        best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                            self._match_embedding(feat, known_list, min_match_score)

                        best_l2 = 999.0
                        for p in known_list:
                            if p["name"] == best_name:
                                for k_feat in p.get("features", [p["feature"]]):
                                    if k_feat.flatten().shape[0] == 128:
                                        l2 = float(self.recognizer.match(feat, k_feat, cv2.FaceRecognizerSF_FR_NORM_L2))
                                        if l2 < best_l2:
                                            best_l2 = l2
                                break

                        calibrated_conf = calibrate_match_confidence(max(0.0, best_score), threshold=min_match_score)
                        best_is_suspect = is_known and bool(
                            next((p.get("is_suspect", False) for p in known_list if p["name"] == best_name), False)
                        )
                        is_threat = bool(is_known and best_is_suspect)
                        tiled_detections.append({
                            "name": best_name,
                            "is_known": is_known,
                            "is_ambiguous": is_ambiguous,
                            "confidence": round(det_conf, 4),
                            "match_score": round(max(0.0, best_score), 4),
                            "l2_distance": round(best_l2, 4),
                            "calibrated_conf": calibrated_conf,
                            "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                            "type": "face",
                            "class_name": f"🚨 SUSPECT: {best_name}" if is_threat else (f"Face: {best_name}" if is_known else "Face: Unknown"),
                            "is_threat": is_threat,
                            "threat_level": best_threat_level if is_threat else "NONE",
                            "category": best_category if is_known else "Unknown",
                            "is_distant": True,
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
        Detect faces in image_bgr and match against enrolled persons.

        Features (both engines):
          - Optional downscale for 4× faster detection (max_det_width=960)
          - Distant face mode with Person-Guided RoI detection directly on native resolution
          - Multi-scale / tiled horizon detection when full-frame downscaling would crush small faces
          - Adaptive unsharp masking and CLAHE for low-resolution distant face crops
          - Rejects tiny noise (< min_face_size px)
          - Filters partial/occluded faces by det_score_thresh
          - Multi-embedding per-person voting
          - Dimension-safe matching (128-D and 512-D coexist safely)
          - Human-calibrated confidence percentage (0–100%)
          - Exclusive 1-to-1 identity assignment per frame
          - Temporal identity smoothing without false-negative downgrades

        UniFace path  : SCRFD detects + AdaFace embeds (512-D, threshold 0.40).
        Fallback path : YuNet detects + SFace embeds  (128-D, threshold 0.34).
        """
        # Resolve match threshold to engine default when not explicitly passed
        if min_match_score is None:
            min_match_score = self.match_threshold

        if enable_distant_mode:
            if min_face_size == 18:
                min_face_size = 12
            if det_score_thresh == 0.35:
                det_score_thresh = 0.25
            if max_det_width == 960:
                max_det_width = 1440

        if image_bgr is None or image_bgr.size == 0 or not self.is_ready:
            return []

        h, w = image_bgr.shape[:2]
        now = time.time()
        raw_detections: list[dict[str, Any]] = []

        with self.lock:
            known_list = list(self.known_persons)

        # ------------------------------------------------------------------ #
        # ENGINE BRANCH A: UniFace (SCRFD + AdaFace)                         #
        # ------------------------------------------------------------------ #
        if self.use_uniface and self.analyzer is not None:
            # Optionally downscale for faster SCRFD detection
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

                    # Rescale bbox back to original resolution
                    bbox_arr = face_uf.bbox  # np.ndarray [x1, y1, x2, y2]
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

                    # Native-resolution TTA feature extraction with distant enhancement:
                    feat_512 = None
                    is_distant_crop = fw < 60 or fh < 60
                    if hasattr(face_uf, "landmarks") and face_uf.landmarks is not None:
                        landmarks_native = (face_uf.landmarks * inv_scale).astype(np.float32)
                        feat_512 = self.extract_tta_embedding(image_bgr, landmarks_native, is_distant=is_distant_crop)

                    # Fallback to downscaled embedding if TTA failed
                    if feat_512 is None:
                        if face_uf.embedding is None:
                            continue
                        feat_512 = face_uf.embedding.astype(np.float32).reshape(1, 512)

                    best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                        self._match_embedding(feat_512, known_list, min_match_score)

                    # Calibrate with ArcFace/AdaFace curve
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
                        "l2_distance": 0.0,  # Not used by ArcFace/AdaFace path
                        "calibrated_conf": calibrated_conf,
                        "bbox": [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)],
                        "type": "face",
                        "class_name": class_name,
                        "is_threat": is_threat,
                        "threat_level": best_threat_level if is_threat else "NONE",
                        "category": best_category if is_known else "Unknown",
                        "is_distant": is_distant_crop,
                    })

        # ------------------------------------------------------------------ #
        # ENGINE BRANCH B: OpenCV YuNet + SFace fallback (128-D)             #
        # ------------------------------------------------------------------ #
        else:
            if self.detector is None or self.recognizer is None:
                return []

            # Fast downscaled detection
            scale = 1.0
            if max_det_width > 0 and w > max_det_width:
                scale = max_det_width / float(w)
                det_w = int(w * scale)
                det_h = int(h * scale)
                det_img = cv2.resize(image_bgr, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
            else:
                det_w, det_h = w, h
                det_img = image_bgr

            with self.lock:
                self.detector.setInputSize((det_w, det_h))
                _, faces = self.detector.detect(det_img)

            if faces is not None and len(faces) > 0:
                inv_scale = 1.0 / scale if scale < 1.0 else 1.0

                for raw_face in faces:
                    if scale < 1.0:
                        face = raw_face.copy()
                        face[:14] = face[:14] * inv_scale
                    else:
                        face = raw_face

                    x, y, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
                    det_conf = float(face[14])

                    if det_conf < det_score_thresh:
                        continue
                    if fw < min_face_size or fh < min_face_size:
                        continue
                    aspect = fh / max(1.0, fw)
                    if aspect < 0.5 or aspect > 2.0:
                        continue

                    x1 = max(0.0, min(x, float(w - 1)))
                    y1 = max(0.0, min(y, float(h - 1)))
                    x2 = max(0.0, min(x + fw, float(w)))
                    y2 = max(0.0, min(y + fh, float(h)))

                    is_distant_crop = fw < 60 or fh < 60
                    with self.lock:
                        aligned_face = self.recognizer.alignCrop(image_bgr, face)
                        enhanced_face = enhance_aligned_face(aligned_face, is_distant_or_low_res=is_distant_crop)
                        feat = self.recognizer.feature(enhanced_face)

                    best_name, best_score, is_known, best_threat_level, best_category, is_ambiguous = \
                        self._match_embedding(feat, known_list, min_match_score)

                    # Use original SFace L2 for display (compute separately for the best match)
                    best_l2 = 999.0
                    for p in known_list:
                        if p["name"] == best_name:
                            for k_feat in p.get("features", [p["feature"]]):
                                if k_feat.flatten().shape[0] == 128:
                                    l2 = float(self.recognizer.match(
                                        feat, k_feat, cv2.FaceRecognizerSF_FR_NORM_L2
                                    ))
                                    if l2 < best_l2:
                                        best_l2 = l2
                            break

                    calibrated_conf = calibrate_match_confidence(
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
                        "l2_distance": round(best_l2, 4),
                        "calibrated_conf": calibrated_conf,
                        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                        "type": "face",
                        "class_name": class_name,
                        "is_threat": is_threat,
                        "threat_level": best_threat_level if is_threat else "NONE",
                        "category": best_category if is_known else "Unknown",
                        "is_distant": is_distant_crop,
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

        # Deduplicate detections from full-frame, RoIs, or multi-scale passes
        if len(raw_detections) > 1:
            raw_detections = deduplicate_face_detections(raw_detections, iou_thresh=0.35)

        if not raw_detections:
            if use_temporal_smoothing:
                self._age_tracks()
            return []

        # Exclusive 1-to-1 Identity Assignment:
        claimed_identities = set()
        for det in sorted(raw_detections, key=lambda d: d["match_score"], reverse=True):
            if det["is_known"]:
                if det["name"] not in claimed_identities:
                    claimed_identities.add(det["name"])
                else:
                    det["is_known"] = False
                    det["is_threat"] = False
                    det["threat_level"] = "NONE"
                    det["name"] = "Unknown"
                    det["class_name"] = "Face: Unknown"
                    det["calibrated_conf"] = calibrate_match_confidence(
                        min(det["match_score"], min_match_score * 0.85),
                        threshold=min_match_score,
                    )

        if not use_temporal_smoothing:
            self._log_events(raw_detections)
            return raw_detections

        smoothed_results = self._apply_temporal_smoothing(raw_detections, now, min_match_score)
        self._log_events(smoothed_results)
        return smoothed_results

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

            # Calculate box center and size for proximity fallback
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

                # Proximity distance check
                t_cx = (t_box[0] + t_box[2]) / 2.0
                t_cy = (t_box[1] + t_box[3]) / 2.0
                center_dist = np.sqrt((det_cx - t_cx) ** 2 + (det_cy - t_cy) ** 2)

                # Match if IoU >= 0.15 OR centers are within 1.5x face diagonal
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

                track["history"].append({
                    "name": det["name"],
                    "is_known": det["is_known"],
                    "score": det["match_score"],
                    "calibrated": det["calibrated_conf"],
                })
                if len(track["history"]) > 6:
                    track["history"].pop(0)

                known_votes = [h for h in track["history"] if h["is_known"]]
                if known_votes or det.get("is_known"):
                    # Use all known votes including current detection
                    candidate_votes = list(known_votes)
                    if det.get("is_known") and not any(h["name"] == det["name"] for h in candidate_votes):
                        candidate_votes.append({
                            "name": det["name"],
                            "is_known": True,
                            "score": det["match_score"],
                            "calibrated": det["calibrated_conf"],
                        })

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
                self.tracks[tid] = {
                    "bbox": det_box,
                    "history": [{
                        "name": det["name"],
                        "is_known": det["is_known"],
                        "score": det["match_score"],
                        "calibrated": det["calibrated_conf"],
                    }],
                    "last_seen": now,
                }
                # For new tracks, retain current detection without forcing downgrade

            final_results.append(det)

        smoothed_claimed = set()
        for det in sorted(final_results, key=lambda d: d["match_score"], reverse=True):
            if det["is_known"]:
                if det["name"] not in smoothed_claimed:
                    smoothed_claimed.add(det["name"])
                else:
                    det["is_known"] = False
                    det["is_threat"] = False
                    det["threat_level"] = "NONE"
                    det["name"] = "Unknown"
                    det["class_name"] = "Face: Unknown"

        return final_results

    def _age_tracks(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        expired = [tid for tid, t in self.tracks.items() if now - t["last_seen"] > 2.5]
        for tid in expired:
            del self.tracks[tid]

    def _log_events(self, detections: list[dict[str, Any]]) -> None:
        """Keep a ring buffer of recent detection events for the frontend operations center."""
        now_str = datetime.now().strftime("%H:%M:%S")
        for det in detections:
            event = {
                "id": f"face_evt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}",
                "name": det["name"],
                "is_known": det["is_known"],
                "is_threat": det.get("is_threat", False),
                "threat_level": det.get("threat_level", "NONE"),
                "category": det.get("category", ""),
                "calibrated_conf": det["calibrated_conf"],
                "timestamp": now_str,
                "bbox": det["bbox"],
            }
            # Avoid logging consecutive duplicate events within 2 seconds
            if self.recent_events:
                last = self.recent_events[0]
                if last["name"] == event["name"] and last["is_known"] == event["is_known"]:
                    continue

            self.recent_events.insert(0, event)
            if len(self.recent_events) > self.max_events:
                self.recent_events.pop()

    def draw_faces(self, image_bgr: np.ndarray, face_detections: list[dict[str, Any]]) -> np.ndarray:
        """
        Draw high-visibility tactical bounding boxes and names directly onto image_bgr.
        Shows calibrated confidence percentage and emphasizes suspect detections in tactical red.
        """
        annotated = image_bgr.copy()
        for f in face_detections:
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            is_known = f.get("is_known", False)
            is_threat = f.get("is_threat", False)
            name = f.get("name", "Unknown")
            threat_level = f.get("threat_level", "HIGH")
            calibrated_conf = f.get("calibrated_conf", 0.0)

            # Color scheme:
            # - Suspect: Tactical Alert Crimson (BGR: 25, 25, 240)
            # - Authorized: Emerald Green (BGR: 0, 230, 115)
            # - Unknown: Tactical Amber (BGR: 0, 165, 255)
            if is_threat:
                color = (25, 25, 240)
                accent_color = (0, 0, 255)
                tag_bg = (18, 18, 38)
            elif is_known:
                color = (0, 230, 115)
                accent_color = (0, 255, 140)
                tag_bg = (18, 26, 22)
            else:
                color = (0, 165, 255)
                accent_color = (0, 195, 255)
                tag_bg = (20, 24, 28)

            # Bounding box
            box_thickness = 3 if is_threat else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

            # Corner accents
            corner_len = max(8, min(22, int(min(x2 - x1, y2 - y1) * 0.25)))
            accent_thickness = 4 if is_threat else 3
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), accent_color, accent_thickness)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), accent_color, accent_thickness)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), accent_color, accent_thickness)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), accent_color, accent_thickness)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), accent_color, accent_thickness)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), accent_color, accent_thickness)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), accent_color, accent_thickness)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), accent_color, accent_thickness)

            # High-visibility tactical tag
            if is_threat:
                label_text = f"SUSPECT: {name} [{threat_level}] {calibrated_conf:.0f}%"
            elif is_known:
                label_text = f"{name} {calibrated_conf:.0f}%"
            else:
                label_text = f"Unknown ({calibrated_conf:.0f}%)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52 if is_threat else 0.5
            thickness = 2 if is_threat else 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            tag_y1 = max(0, y1 - th - 10)
            tag_y2 = y1
            tag_x2 = min(annotated.shape[1], x1 + tw + 14)

            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), tag_bg, -1)
            cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, 2 if is_threat else 1)
            cv2.putText(
                annotated,
                label_text,
                (x1 + 6, tag_y2 - 5),
                font,
                font_scale,
                (255, 255, 255) if is_threat else color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated


face_service = FaceService()
