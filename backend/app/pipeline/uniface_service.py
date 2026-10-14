from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Optional
import uuid

import cv2
import numpy as np

from app.core.config import settings

# UniFace core imports
from uniface import FaceAnalyzer
from uniface.constants import ArcFaceWeights, MiniFASNetWeights, SCRFDWeights
from uniface.detection import SCRFD
from uniface.recognition import ArcFace
from uniface.attribute import FairFace, FaceAttribNet
from uniface.spoofing import MiniFASNet
from uniface.privacy import BlurFace
from uniface.face_utils import compute_similarity
from uniface.stores import FAISS

logger = logging.getLogger("SIH26187.UniFaceService")


def calibrate_arcface_confidence(cosine_score: float, threshold: float = 0.42) -> float:
    """
    Calibrate raw 512-D ArcFace cosine similarity (-1.0 to 1.0) into a human-friendly
    confidence percentage (0% to 100%).
    
    ArcFace Decision Boundaries (calibrated for CCTV / surveillance):
      - c < 0.20: Non-match (noise / random face) -> 0% - 15%
      - c < threshold (0.42): Unrecognized / below threshold -> 15% - 49%
      - c = threshold: Standard decision boundary -> 50%
      - c = 0.50: Strong match -> 78%
      - c = 0.60: Verified high confidence -> 92%
      - c >= 0.70: Near identical match -> 98% - 99.9%
    """
    c = float(cosine_score)
    th = float(threshold)
    if c <= 0.0:
        return 0.0
    elif c <= 0.20:
        val = (c / 0.20) * 15.0
    elif c < th:
        val = 15.0 + ((c - 0.20) / max(0.01, th - 0.20)) * 34.0
    elif c <= 0.52:
        val = 50.0 + ((c - th) / max(0.01, 0.52 - th)) * 32.0
    elif c <= 0.65:
        val = 82.0 + ((c - 0.52) / (0.65 - 0.52)) * 13.0
    else:
        val = 95.0 + min(1.0, (c - 0.65) / 0.20) * 4.9

    return round(float(np.clip(val, 0.0, 99.9)), 1)


class UniFaceService:
    """
    High-Performance Unified Biometric Surveillance Service for MayaJaal.
    Leverages UniFace:
      - SCRFD: High-recall face detector for small & distant CCTV faces
      - ArcFace: 512-dimensional angular margin embeddings
      - FairFace: Demographics (Age Group, Gender, Race)
      - FaceAttribNet: Disguise & Attribute detection (Mask, Eyeglasses, Sunglasses)
      - MiniFASNet: Anti-spoofing / Presentation Attack liveness verification
      - BlurFace: Selective privacy-preserving bystander anonymization
      - FAISS: Sub-millisecond vector similarity search across large suspect databases
    """

    def __init__(self) -> None:
        self.data_dir = settings.model_directory.parent / "app" / "data" / "faces"
        self.thumbnails_dir = self.data_dir / "thumbnails"
        self.db_file = self.data_dir / "known_faces.json"
        self.faiss_dir = self.data_dir / "faiss_index"

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.analyzer: Optional[FaceAnalyzer] = None
        self.spoofer: Optional[MiniFASNet] = None
        self.blur_tool: Optional[BlurFace] = None
        self.faiss_store: Optional[FAISS] = None

        # In-memory enrolled persons with 512-D embeddings
        self.known_persons: list[dict[str, Any]] = []

        # Recent detection events
        self.recent_events: list[dict[str, Any]] = []
        self.max_events: int = 50

        self._init_models()
        self.load_database()

    def _init_models(self) -> None:
        """Initialize UniFace components and FAISS vector store with graceful fallback."""
        try:
            logger.info("Initializing UniFace AI models (SCRFD, ArcFace, FairFace, FaceAttribNet, MiniFASNet)...")
            detector = SCRFD(model_name=SCRFDWeights.SCRFD_500M_KPS)
            recognizer = ArcFace(model_name=ArcFaceWeights.MNET)
            fairface = FairFace()
            attribnet = FaceAttribNet()

            self.analyzer = FaceAnalyzer(
                detector=detector,
                recognizer=recognizer,
                predictors=[fairface, attribnet],
            )
            self.spoofer = MiniFASNet(model_name=MiniFASNetWeights.V2)
            self.blur_tool = BlurFace(method="gaussian", blur_strength=3.0)

            try:
                self.faiss_store = FAISS(embedding_size=512, db_path=str(self.faiss_dir))
                logger.info("UniFace FAISS vector store initialized at %s", self.faiss_dir)
            except Exception as f_err:
                logger.warning("Could not initialize FAISS store: %s", f_err)
                self.faiss_store = None

            logger.info("UniFace models initialized successfully.")
        except Exception as exc:
            logger.exception("Failed to initialize UniFace models: %s", exc)

    def load_database(self) -> None:
        """Load enrolled persons from disk, ensure 512-D ArcFace features exist, and sync FAISS index."""
        with self.lock:
            self.known_persons = []
            if not self.db_file.exists():
                return

            try:
                with open(self.db_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

                dirty = False
                for r in records:
                    person_id = r.get("id")
                    name = r.get("name", "Unknown")
                    thumb_name = f"{person_id}.jpg"
                    image_url = f"/api/faces/thumbnail/{thumb_name}"
                    is_suspect = bool(r.get("is_suspect", False))
                    threat_level = str(r.get("threat_level", "HIGH" if is_suspect else "LOW"))
                    category = str(r.get("category", "Wanted / BOLO" if is_suspect else "Authorized Personnel"))
                    notes = str(r.get("notes", ""))

                    features_512: list[np.ndarray] = []
                    raw_512 = r.get("features_512", [])
                    if isinstance(raw_512, list) and len(raw_512) > 0:
                        for item in raw_512:
                            if isinstance(item, list) and len(item) == 512:
                                features_512.append(np.array(item, dtype=np.float32))

                    # If missing 512-D features, attempt extraction from saved thumbnail
                    if len(features_512) == 0 and self.analyzer and self.analyzer.recognizer:
                        thumb_file = self.thumbnails_dir / thumb_name
                        if thumb_file.exists():
                            thumb_img = cv2.imread(str(thumb_file))
                            if thumb_img is not None:
                                try:
                                    faces = self.analyzer.detector.detect(thumb_img)
                                    if faces:
                                        emb = self.analyzer.recognizer.get_normalized_embedding(thumb_img, faces[0].landmarks)
                                        features_512.append(emb)
                                        r["features_512"] = [emb.tolist()]
                                        dirty = True
                                        logger.info("Extracted 512-D ArcFace embedding for %s from thumbnail", name)
                                except Exception as exc:
                                    logger.warning("Could not extract ArcFace embedding from thumbnail for %s: %s", name, exc)

                    self.known_persons.append({
                        "id": person_id,
                        "name": name,
                        "image_url": image_url,
                        "created_at": r.get("created_at", ""),
                        "features_512": features_512,
                        "is_suspect": is_suspect,
                        "threat_level": threat_level,
                        "category": category,
                        "notes": notes,
                    })

                if dirty:
                    self._save_database()

                # Synchronize FAISS Vector Store
                if self.faiss_store is not None:
                    try:
                        self.faiss_store = FAISS(embedding_size=512, db_path=str(self.faiss_dir))
                        v_count = 0
                        for p in self.known_persons:
                            p_meta = {
                                "person_id": p["id"],
                                "name": p["name"],
                                "is_suspect": p.get("is_suspect", False),
                                "threat_level": p.get("threat_level", "HIGH" if p.get("is_suspect") else "LOW"),
                                "category": p.get("category", "Wanted / BOLO" if p.get("is_suspect") else "Authorized Personnel"),
                                "notes": p.get("notes", ""),
                                "image_url": p.get("image_url", ""),
                            }
                            for f in p.get("features_512", []):
                                f_arr = np.asarray(f, dtype=np.float32).ravel()
                                norm = float(np.linalg.norm(f_arr))
                                if norm > 1e-6:
                                    self.faiss_store.add(f_arr / norm, p_meta)
                                    v_count += 1
                        if v_count > 0:
                            self.faiss_store.save()
                        logger.info("UniFaceService: FAISS store populated with %d vectors across %d identities.",
                                    v_count, len(self.known_persons))
                    except Exception as f_sync_err:
                        logger.warning("UniFaceService: FAISS store synchronization warning: %s", f_sync_err)

                logger.info("UniFaceService: Loaded %d enrolled persons (%d with 512-D embeddings).",
                            len(self.known_persons),
                            sum(1 for p in self.known_persons if len(p.get("features_512", [])) > 0))
            except Exception as exc:
                logger.exception("Error loading face database in UniFaceService: %s", exc)

    def _save_database(self) -> None:
        """Persist enrolled database to disk preserving all schema fields."""
        if not self.db_file.exists():
            return
        try:
            with open(self.db_file, "r", encoding="utf-8") as f:
                existing_records = {r.get("id"): r for r in json.load(f)}
        except Exception:
            existing_records = {}

        for p in self.known_persons:
            pid = p["id"]
            rec = existing_records.get(pid, {
                "id": pid,
                "name": p["name"],
                "image_url": p["image_url"],
                "created_at": p.get("created_at", ""),
            })
            rec["is_suspect"] = p.get("is_suspect", False)
            rec["threat_level"] = p.get("threat_level", "HIGH")
            rec["category"] = p.get("category", "Wanted / BOLO")
            rec["notes"] = p.get("notes", "")
            if p.get("features_512"):
                rec["features_512"] = [f.tolist() for f in p["features_512"]]
            existing_records[pid] = rec

        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(list(existing_records.values()), f, indent=2)

    def is_ready(self) -> bool:
        return self.analyzer is not None and self.analyzer.detector is not None

    def search_suspect_embedding(
        self,
        embedding: np.ndarray,
        threshold: float = 0.42,
    ) -> tuple[Optional[dict[str, Any]], float]:
        """
        Perform sub-millisecond similarity search across suspect database using FAISS IndexFlatIP.
        Returns (metadata_dict, similarity_score). Falls back to linear scan if FAISS is uninitialized.
        """
        if embedding is None:
            return None, 0.0

        f_arr = np.asarray(embedding, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(f_arr))
        if norm > 1e-6:
            f_arr = f_arr / norm

        # 1. Primary: Sub-millisecond FAISS Vector Store Search
        if self.faiss_store is not None and self.faiss_store.size > 0:
            try:
                meta, score = self.faiss_store.search(f_arr, threshold=threshold)
                if meta is not None:
                    return dict(meta), float(score)
                return None, float(score)
            except Exception as exc:
                logger.debug("FAISS search exception, falling back to linear: %s", exc)

        # 2. Fallback: Linear search across known_persons in memory
        with self.lock:
            best_meta: Optional[dict[str, Any]] = None
            best_score: float = -1.0
            for p in self.known_persons:
                for k_emb in p.get("features_512", []):
                    sc = float(compute_similarity(f_arr, k_emb))
                    if sc > best_score:
                        best_score = sc
                        if sc >= threshold:
                            best_meta = {
                                "person_id": p["id"],
                                "name": p["name"],
                                "is_suspect": p.get("is_suspect", False),
                                "threat_level": p.get("threat_level", "HIGH" if p.get("is_suspect") else "LOW"),
                                "category": p.get("category", "Wanted / BOLO" if p.get("is_suspect") else "Authorized Personnel"),
                                "notes": p.get("notes", ""),
                                "image_url": p.get("image_url", ""),
                            }
            return best_meta, max(0.0, best_score)

    def extract_face_from_person_crop(
        self,
        person_crop_bgr: np.ndarray,
        min_face_size: int = 24,
        verify_liveness: bool = True,
    ) -> Optional[dict[str, Any]]:
        """
        Extract 512-D ArcFace face embedding and biometric attributes directly from a full-body person crop.
        Enables seamless Face-to-Body Multi-Camera ReID Fusion (Cross-Modal Tracking).
        """
        if person_crop_bgr is None or person_crop_bgr.size == 0 or self.analyzer is None:
            return None

        h, w = person_crop_bgr.shape[:2]
        if h < min_face_size or w < min_face_size:
            return None

        # CCTV person crops: head/face is typically in the upper 55% of the bounding box
        search_h = int(h * 0.55)
        upper_crop = person_crop_bgr[:max(min_face_size, search_h), :]

        try:
            # 1. Search upper body crop first for speed
            faces = self.analyzer.analyze(upper_crop)
            analyzed_source = upper_crop
            if not faces:
                # Fallback to entire person crop if head wasn't in top 55% (e.g., crouching or sitting)
                faces = self.analyzer.analyze(person_crop_bgr)
                analyzed_source = person_crop_bgr

            if not faces:
                return None

            # Pick largest face
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if best_face.embedding is None:
                return None

            emb = np.asarray(best_face.embedding, dtype=np.float32).ravel()
            norm = float(np.linalg.norm(emb))
            if norm > 1e-6:
                emb = emb / norm

            # 2. Liveness check
            is_real = True
            liveness_conf = 1.0
            if verify_liveness and self.spoofer is not None:
                try:
                    sp_res = self.spoofer.predict(analyzed_source, best_face.bbox)
                    is_real = bool(sp_res.is_real)
                    liveness_conf = round(float(sp_res.confidence), 3)
                except Exception:
                    pass

            # 3. Disguise / attributes
            mask_prob = float(best_face.mask) if best_face.mask is not None else 0.0
            sunglasses_prob = float(best_face.sunglasses) if best_face.sunglasses is not None else 0.0
            eyeglasses_prob = float(best_face.eyeglasses) if best_face.eyeglasses is not None else 0.0

            disguise_labels = []
            if mask_prob >= 0.5:
                disguise_labels.append("Face Mask")
            if sunglasses_prob >= 0.5:
                disguise_labels.append("Sunglasses")

            # 4. Instant FAISS suspect search
            suspect_meta, score = self.search_suspect_embedding(emb, threshold=0.42)
            is_suspect_match = (suspect_meta is not None) and bool(suspect_meta.get("is_suspect", False))

            return {
                "embedding": emb,
                "confidence": float(best_face.confidence),
                "is_match": suspect_meta is not None,
                "match_score": round(score, 4),
                "suspect_meta": suspect_meta,
                "is_suspect": is_suspect_match,
                "suspect_name": suspect_meta.get("name") if suspect_meta else None,
                "threat_level": suspect_meta.get("threat_level") if suspect_meta else None,
                "category": suspect_meta.get("category") if suspect_meta else None,
                "demographics": {
                    "gender": best_face.sex if best_face.gender is not None else "Unknown",
                    "age_group": best_face.age_group or "Unknown",
                    "race": best_face.race or "Unknown",
                },
                "attributes": {
                    "has_disguise": len(disguise_labels) > 0,
                    "disguise_labels": disguise_labels,
                    "mask_prob": round(mask_prob, 3),
                    "sunglasses_prob": round(sunglasses_prob, 3),
                    "eyeglasses_prob": round(eyeglasses_prob, 3),
                },
                "liveness": {
                    "is_real": is_real,
                    "confidence": liveness_conf,
                },
            }
        except Exception as exc:
            logger.debug("Face extraction from person crop failed: %s", exc)
            return None

    def analyze_frame(
        self,
        image_bgr: np.ndarray,
        min_match_score: float = 0.42,
        min_face_size: int = 24,
        verify_liveness: bool = True,
        anonymize_bystanders: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Run end-to-end multi-attribute biometric analysis:
          1. Detect faces & extract 512-D embeddings
          2. Classify demographics (FairFace)
          3. Detect disguise / face state (FaceAttribNet: mask, glasses, sunglasses)
          4. Run Anti-Spoofing Liveness check (MiniFASNet)
          5. Match against BOLO / watchlist database using FAISS vector search
          6. Optionally anonymize bystanders
        """
        if image_bgr is None or image_bgr.size == 0 or self.analyzer is None:
            return []

        h, w = image_bgr.shape[:2]

        with self.lock:
            try:
                faces = self.analyzer.analyze(image_bgr)
            except Exception as exc:
                logger.error("UniFace analyzer failed: %s", exc)
                return []

        if not faces:
            return []

        raw_detections: list[dict[str, Any]] = []
        bystander_bboxes: list[list[int]] = []

        for face in faces:
            bbox = face.bbox
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fw = max(1, x2 - x1)
            fh = max(1, y2 - y1)

            # Filter out tiny noise
            if fw < min_face_size or fh < min_face_size:
                continue

            # Clamp coordinates
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # 1. Anti-Spoofing / Liveness Check
            is_real = True
            liveness_conf = 1.0
            if verify_liveness and self.spoofer is not None:
                try:
                    spoof_res = self.spoofer.predict(image_bgr, face.bbox)
                    is_real = bool(spoof_res.is_real)
                    liveness_conf = round(float(spoof_res.confidence), 3)
                except Exception as exc:
                    logger.debug("Anti-spoofing check error: %s", exc)

            # 2. Identification Match vs Enrolled Database (ArcFace 512-D with FAISS acceleration)
            best_name = "Unknown"
            best_score = -1.0
            is_known = False
            best_is_suspect = False
            best_threat_level = "LOW"
            best_category = "Unknown"

            if face.embedding is not None:
                matched_meta, score = self.search_suspect_embedding(face.embedding, threshold=min_match_score)
                best_score = score
                if matched_meta is not None:
                    best_name = matched_meta.get("name", "Unknown")
                    is_known = True
                    best_is_suspect = bool(matched_meta.get("is_suspect", False))
                    best_threat_level = matched_meta.get("threat_level", "HIGH" if best_is_suspect else "LOW")
                    best_category = matched_meta.get("category", "Wanted / BOLO" if best_is_suspect else "Authorized Personnel")

            calibrated_conf = calibrate_arcface_confidence(max(0.0, best_score), threshold=min_match_score)
            is_threat = bool(is_known and best_is_suspect)

            # 3. Disguise / Attribute Detection
            mask_prob = float(face.mask) if face.mask is not None else 0.0
            sunglasses_prob = float(face.sunglasses) if face.sunglasses is not None else 0.0
            eyeglasses_prob = float(face.eyeglasses) if face.eyeglasses is not None else 0.0
            
            disguise_labels = []
            if mask_prob >= 0.5:
                disguise_labels.append("Face Mask")
            if sunglasses_prob >= 0.5:
                disguise_labels.append("Sunglasses")

            has_disguise = len(disguise_labels) > 0

            # 4. Demographics
            demographics = {
                "gender": face.sex if face.gender is not None else "Unknown",
                "age_group": face.age_group or "Unknown",
                "race": face.race or "Unknown",
            }

            # Class label formatting
            if is_threat:
                class_name = f"🚨 SUSPECT: {best_name}"
            elif is_known:
                class_name = f"Face: {best_name}"
            elif not is_real:
                class_name = "⚠️ FAKE / SPOOF FACE"
            elif has_disguise:
                class_name = f"Face: Disguised ({', '.join(disguise_labels)})"
            else:
                class_name = "Face: Unknown"

            det_entry = {
                "name": best_name,
                "is_known": is_known,
                "confidence": round(float(face.confidence), 4),
                "match_score": round(max(0.0, best_score), 4),
                "calibrated_conf": calibrated_conf,
                "bbox": [x1, y1, x2, y2],
                "type": "face",
                "class_name": class_name,
                "is_threat": is_threat,
                "threat_level": best_threat_level if is_threat else "NONE",
                "category": best_category if is_known else "Unknown",
                "engine": "uniface_arcface",
                "demographics": demographics,
                "attributes": {
                    "mask_prob": round(mask_prob, 3),
                    "sunglasses_prob": round(sunglasses_prob, 3),
                    "eyeglasses_prob": round(eyeglasses_prob, 3),
                    "has_disguise": has_disguise,
                    "disguise_labels": disguise_labels,
                },
                "liveness": {
                    "is_real": is_real,
                    "confidence": liveness_conf,
                },
            }

            raw_detections.append(det_entry)

            # If bystander anonymization requested, collect non-threat bounding boxes
            if anonymize_bystanders and not is_threat:
                bystander_bboxes.append([x1, y1, x2, y2])

        # Apply selective privacy anonymization
        if anonymize_bystanders and bystander_bboxes and self.blur_tool is not None:
            self.blur_tool(image_bgr, bystander_bboxes, inplace=True)

        self._log_events(raw_detections)
        return raw_detections

    def _log_events(self, detections: list[dict[str, Any]]) -> None:
        """Maintain telemetry history of significant face detections."""
        now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for det in detections:
            # Only log identified persons or flagged threats or spoofing attempts
            if det["is_known"] or det["is_threat"] or not det["liveness"]["is_real"] or det["attributes"]["has_disguise"]:
                self.recent_events.insert(0, {
                    "id": str(uuid.uuid4())[:8],
                    "name": det["name"],
                    "is_known": det["is_known"],
                    "calibrated_conf": det["calibrated_conf"],
                    "timestamp": now_iso,
                    "bbox": det["bbox"],
                    "is_threat": det["is_threat"],
                    "threat_level": det["threat_level"],
                    "category": det["category"],
                    "demographics": det["demographics"],
                    "liveness": det["liveness"],
                    "attributes": det["attributes"],
                    "engine": "uniface",
                })
        self.recent_events = self.recent_events[:self.max_events]

    def register_face(
        self,
        image_bgr: np.ndarray,
        name: str,
        is_suspect: bool = True,
        threat_level: str = "HIGH",
        category: str = "Wanted / BOLO",
        notes: str = "",
    ) -> dict[str, Any]:
        """Enroll suspect directly with UniFace 512-D ArcFace feature."""
        name = name.strip()
        if not name:
            return {"success": False, "error": "Name cannot be blank"}
        if image_bgr is None or image_bgr.size == 0 or self.analyzer is None:
            return {"success": False, "error": "Invalid image or UniFace not initialized"}

        with self.lock:
            faces = self.analyzer.analyze(image_bgr)

        if not faces:
            return {"success": False, "error": "No face detected in this image."}

        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if best_face.embedding is None:
            return {"success": False, "error": "Could not extract ArcFace embedding."}

        x1, y1, x2, y2 = [int(v) for v in best_face.bbox]
        h, w = image_bgr.shape[:2]
        crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

        with self.lock:
            existing = next((p for p in self.known_persons if p["name"].lower() == name.lower()), None)
            emb = best_face.embedding

            if existing:
                if "features_512" not in existing:
                    existing["features_512"] = []
                existing["features_512"].append(emb)
                existing["is_suspect"] = bool(is_suspect)
                existing["threat_level"] = str(threat_level).upper()
                existing["category"] = str(category).strip() or existing.get("category", "Wanted / BOLO")
                if notes:
                    existing["notes"] = str(notes).strip()

                thumb_path = self.thumbnails_dir / f"{existing['id']}.jpg"
                if crop.size > 0:
                    cv2.imwrite(str(thumb_path), crop)

                if self.faiss_store is not None:
                    try:
                        f_arr = np.asarray(emb, dtype=np.float32).ravel()
                        norm = float(np.linalg.norm(f_arr))
                        if norm > 1e-6:
                            meta = {
                                "person_id": existing["id"],
                                "name": existing["name"],
                                "is_suspect": existing["is_suspect"],
                                "threat_level": existing["threat_level"],
                                "category": existing["category"],
                                "notes": existing.get("notes", ""),
                                "image_url": existing["image_url"],
                            }
                            self.faiss_store.add(f_arr / norm, meta)
                            self.faiss_store.save()
                    except Exception as f_err:
                        logger.warning("FAISS store update error: %s", f_err)

                self._save_database()
                return {
                    "success": True,
                    "person": {
                        "id": existing["id"],
                        "name": existing["name"],
                        "image_url": existing["image_url"],
                        "created_at": existing["created_at"],
                        "sample_count": len(existing["features_512"]),
                        "is_suspect": existing["is_suspect"],
                        "threat_level": existing["threat_level"],
                        "category": existing["category"],
                        "notes": existing.get("notes", ""),
                    },
                    "message": f"Added ArcFace sample to profile for {existing['name']}!",
                }
            else:
                person_id = f"person_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                thumb_filename = f"{person_id}.jpg"
                thumb_path = self.thumbnails_dir / thumb_filename
                if crop.size > 0:
                    cv2.imwrite(str(thumb_path), crop)

                new_record = {
                    "id": person_id,
                    "name": name,
                    "image_url": f"/api/faces/thumbnail/{thumb_filename}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "features_512": [emb],
                    "is_suspect": bool(is_suspect),
                    "threat_level": str(threat_level).upper(),
                    "category": str(category).strip() or "Wanted / BOLO",
                    "notes": str(notes).strip(),
                }
                self.known_persons.append(new_record)

                if self.faiss_store is not None:
                    try:
                        f_arr = np.asarray(emb, dtype=np.float32).ravel()
                        norm = float(np.linalg.norm(f_arr))
                        if norm > 1e-6:
                            meta = {
                                "person_id": new_record["id"],
                                "name": new_record["name"],
                                "is_suspect": new_record["is_suspect"],
                                "threat_level": new_record["threat_level"],
                                "category": new_record["category"],
                                "notes": new_record["notes"],
                                "image_url": new_record["image_url"],
                            }
                            self.faiss_store.add(f_arr / norm, meta)
                            self.faiss_store.save()
                    except Exception as f_err:
                        logger.warning("FAISS store add error: %s", f_err)

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
                    "message": f"Successfully enrolled face profile for {name} with UniFace!",
                }

    def delete_person(self, person_id: str) -> dict[str, Any]:
        """Remove person from enrolled database, remove thumbnails, and remove from FAISS index."""
        with self.lock:
            prev_len = len(self.known_persons)
            self.known_persons = [p for p in self.known_persons if p["id"] != person_id]
            if len(self.known_persons) < prev_len:
                self._save_database()
                if self.faiss_store is not None:
                    try:
                        self.faiss_store.remove("person_id", person_id)
                        self.faiss_store.save()
                    except Exception as f_err:
                        logger.warning("FAISS store remove error: %s", f_err)

                thumb_path = self.thumbnails_dir / f"{person_id}.jpg"
                if thumb_path.exists():
                    try:
                        thumb_path.unlink()
                    except Exception:
                        pass
                return {"success": True, "message": f"Successfully deleted person {person_id}"}
            return {"success": False, "error": "Person ID not found"}

    def draw_faces(self, image_bgr: np.ndarray, face_detections: list[dict[str, Any]]) -> np.ndarray:
        """
        Draw rich tactical HUD overlays on detected faces:
        - Bounding boxes + corner accents
        - Identity & match percentage
        - Liveness status (🛡️ LIVE vs ⚠️ FAKE / SPOOF)
        - Demographics tag (Gender, Age Group)
        - Disguise tag (Mask, Sunglasses)
        """
        annotated = image_bgr.copy()
        for f in face_detections:
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            is_known = f.get("is_known", False)
            is_threat = f.get("is_threat", False)
            name = f.get("name", "Unknown")
            threat_level = f.get("threat_level", "HIGH")
            calibrated_conf = f.get("calibrated_conf", 0.0)
            
            liveness = f.get("liveness", {})
            is_real = liveness.get("is_real", True)
            
            attributes = f.get("attributes", {})
            has_disguise = attributes.get("has_disguise", False)
            disguise_labels = attributes.get("disguise_labels", [])
            
            demographics = f.get("demographics", {})
            gender = demographics.get("gender", "Unknown")
            age_group = demographics.get("age_group", "Unknown")

            # Determine tactical color:
            # Threat: Red (25, 25, 240)
            # Fake/Spoof: Magenta (255, 0, 200)
            # Known/Authorized: Emerald (0, 230, 115)
            # Unknown: Amber (0, 165, 255)
            if not is_real:
                color = (255, 0, 200)
                accent_color = (255, 50, 220)
            elif is_threat:
                color = (25, 25, 240)
                accent_color = (0, 0, 255)
            elif is_known:
                color = (0, 230, 115)
                accent_color = (0, 255, 140)
            else:
                color = (0, 165, 255)
                accent_color = (0, 195, 255)

            # Box and corners
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            corner_len = max(8, min(20, int(min(x2 - x1, y2 - y1) * 0.22)))
            cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), accent_color, 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), accent_color, 3)
            cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), accent_color, 3)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), accent_color, 3)
            cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), accent_color, 3)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), accent_color, 3)
            cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), accent_color, 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), accent_color, 3)

            # Primary Label
            if not is_real:
                label_text = f"SPOOF ALERT! ({liveness.get('confidence', 0)*100:.0f}%)"
            elif is_threat:
                label_text = f"SUSPECT: {name} [{threat_level}] {calibrated_conf:.0f}%"
            elif is_known:
                label_text = f"{name} {calibrated_conf:.0f}%"
            else:
                label_text = f"Unknown ({calibrated_conf:.0f}%)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label_text, font, 0.48, 1)
            tag_y1 = max(0, y1 - th - 8)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 10, y1), (15, 20, 25), -1)
            cv2.rectangle(annotated, (x1, tag_y1), (x1 + tw + 10, y1), color, 1)
            cv2.putText(annotated, label_text, (x1 + 5, y1 - 4), font, 0.48, color, 1, cv2.LINE_AA)

            # Secondary Tag (Demographics & Disguise) under bounding box
            sub_tags = []
            if gender != "Unknown" or age_group != "Unknown":
                g_code = "M" if gender == "Male" else ("F" if gender == "Female" else gender)
                sub_tags.append(f"{g_code}, {age_group}")
            if has_disguise:
                sub_tags.append("/".join(disguise_labels))

            if sub_tags:
                sub_text = " | ".join(sub_tags)
                (stw, sth), _ = cv2.getTextSize(sub_text, font, 0.38, 1)
                sub_y2 = min(image_bgr.shape[0], y2 + sth + 8)
                cv2.rectangle(annotated, (x1, y2), (x1 + stw + 8, sub_y2), (15, 20, 25), -1)
                cv2.rectangle(annotated, (x1, y2), (x1 + stw + 8, sub_y2), (100, 110, 120), 1)
                cv2.putText(annotated, sub_text, (x1 + 4, sub_y2 - 3), font, 0.38, (210, 220, 230), 1, cv2.LINE_AA)

        return annotated


# Global singleton instance
uniface_service = UniFaceService()
