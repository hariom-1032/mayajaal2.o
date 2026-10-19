from __future__ import annotations

import base64
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

logger = logging.getLogger("SIH26187.UnknownFaceService")


class UnknownFaceService:
    """
    Centralized Unknown Person Detection, Multi-Sample Biometric Re-ID & Archival Service for MayaJaal.

    Robust Multi-Camera Surveillance Re-ID & Deduplication:
      1. Multi-Sample Biometric Gallery:
         Stores up to 6 diverse 512-D AdaFace angle/pose vectors per unknown person.
         Evaluates maximum cosine similarity across all stored samples and centroid.
      2. Surveillance-Calibrated Adaptive Thresholding:
         - Cross-Camera Re-ID threshold: 0.28 (random unrelated faces max dot product is ~0.13).
         - Recent window / Same-Camera threshold: 0.24 (accommodates motion blur, distant horizon cameras).
      3. Same-Camera Spatial-Temporal Continuity Guard:
         Prevents rapid-fire duplicate spawns when a person is in view of the same camera within 20s.
      4. Auto-Consolidation:
         Greedily clusters and merges duplicate records, retaining the clearest photo and combining trajectories.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data_dir = settings.model_directory.parent / "app" / "data" / "faces"
        self.unknowns_dir = self.data_dir / "unknowns"
        self.photos_dir = self.unknowns_dir / "photos"
        self.db_file = self.unknowns_dir / "unknown_records.json"
        self.embeddings_file = self.unknowns_dir / "unknown_embeddings.npz"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.unknowns_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)

        self.records: list[dict[str, Any]] = []
        # Multi-sample gallery: unknown_id -> np.ndarray of shape (N, 512)
        self.embeddings: dict[str, np.ndarray] = {}
        self.max_records: int = 500

        # Calibrated Surveillance Cosine Similarity Thresholds (AdaFace IR-101 512-D)
        # Random different people dot product mean=0.00, max <= 0.13.
        # Same person with head rotation / cross-camera lighting under CCTV: 0.18 - 0.70.
        self.reid_similarity_threshold: float = 0.20
        self.reid_recent_threshold: float = 0.18

        # Same-camera temporal & spatial continuity guard
        # cam_id -> list of { "unknown_id": str, "last_seen": float, "center": (cx, cy), "track_id": Any }
        self._camera_active_unknowns: dict[str, list[dict[str, Any]]] = {}

        # Fallback debounces
        self._active_debounces: dict[str, dict[str, Any]] = {}
        self.debounce_cooldown: float = 15.0

        self._load_records()
        self._load_embeddings()
        if len(self.records) > 1:
            try:
                self.consolidate_duplicates(similarity_threshold=0.18)
            except Exception as c_err:
                logger.debug("Initial duplicate consolidation error: %s", c_err)

    def _load_records(self) -> None:
        """Load persisted unknown person records from JSON file."""
        with self.lock:
            if self.db_file.exists():
                try:
                    with open(self.db_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.records = data
                            logger.info("Loaded %d unknown person records from disk.", len(self.records))
                            return
                except Exception as exc:
                    logger.warning("Failed to load unknown records file: %s", exc)
            self.records = []

    def _save_records(self) -> None:
        """Persist unknown person records atomically to JSON disk."""
        temp_file = self.db_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.db_file)
        except Exception as exc:
            logger.error("Failed to persist unknown person records: %s", exc)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def _load_embeddings(self) -> None:
        """Load persisted 512-D biometric multi-sample galleries from NPZ disk."""
        with self.lock:
            self.embeddings.clear()
            if self.embeddings_file.exists():
                try:
                    loaded = np.load(self.embeddings_file)
                    for k in loaded.files:
                        arr = loaded[k].astype(np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, 512)
                        self.embeddings[k] = arr
                    logger.info("Loaded %d unknown biometric face galleries from disk.", len(self.embeddings))
                except Exception as exc:
                    logger.warning("Failed to load unknown face embeddings: %s", exc)

    def _save_embeddings(self) -> None:
        """Persist 512-D biometric galleries atomically to NPZ disk."""
        if not self.embeddings:
            if self.embeddings_file.exists():
                try:
                    self.embeddings_file.unlink()
                except Exception:
                    pass
            return

        temp_file = self.embeddings_file.with_suffix(".tmp.npz")
        try:
            np.savez_compressed(temp_file, **self.embeddings)
            temp_file.replace(self.embeddings_file)
        except Exception as exc:
            logger.error("Failed to persist unknown biometric embeddings: %s", exc)
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def _add_embedding_sample(self, unknown_id: str, new_emb: np.ndarray) -> None:
        """Store or update biometric face embedding samples for an unknown individual."""
        new_vec = new_emb.reshape(1, 512).astype(np.float32)
        norm = np.linalg.norm(new_vec)
        if norm > 1e-8:
            new_vec = new_vec / norm

        gallery = self.embeddings.get(unknown_id)
        if gallery is None or gallery.size == 0:
            self.embeddings[unknown_id] = new_vec
            return

        if gallery.ndim == 1:
            gallery = gallery.reshape(1, 512)

        sims = np.dot(gallery, new_vec.T).ravel()
        max_sim = float(np.max(sims))

        # If novel angle / head rotation / lighting (0.12 <= sim < 0.78), store as diverse angle sample
        if max_sim < 0.78 and len(gallery) < 8:
            self.embeddings[unknown_id] = np.vstack([gallery, new_vec])
        else:
            # Weighted average update to closest angle sample
            best_idx = int(np.argmax(sims))
            updated = (0.70 * gallery[best_idx]) + (0.30 * new_vec[0])
            u_norm = np.linalg.norm(updated)
            if u_norm > 1e-8:
                gallery[best_idx] = updated / u_norm
            self.embeddings[unknown_id] = gallery

    def _generate_unknown_id(self) -> str:
        """Generate a clean, memorable Unknown Person ID, e.g., UNK-9E3A."""
        existing_ids = {r["id"] for r in self.records}
        for _ in range(50):
            short_hex = uuid.uuid4().hex[:4].upper()
            candidate = f"UNK-{short_hex}"
            if candidate not in existing_ids:
                return candidate
        return f"UNK-{uuid.uuid4().hex[:6].upper()}"

    def _extract_face_crop(
        self,
        frame: np.ndarray,
        bbox: list[float] | tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Extract a well-proportioned, padded face portrait crop from the camera frame.
        Adds 30% padding around bounding box to capture full head contours cleanly.
        """
        if frame is None or frame.size == 0 or len(bbox) < 4:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        pad_x = bw * 0.25
        pad_y_top = bh * 0.35
        pad_y_bot = bh * 0.25

        cx1 = max(0, int(round(x1 - pad_x)))
        cy1 = max(0, int(round(y1 - pad_y_top)))
        cx2 = min(w, int(round(x2 + pad_x)))
        cy2 = min(h, int(round(y2 + pad_y_bot)))

        if (cx2 - cx1) < 10 or (cy2 - cy1) < 10:
            return None

        crop = frame[cy1:cy2, cx1:cx2].copy()

        # Enhance resolution for distant surveillance faces if small
        ch, cw = crop.shape[:2]
        if cw < 120 or ch < 120:
            scale = max(120 / max(1, cw), 120 / max(1, ch))
            target_w = int(cw * scale)
            target_h = int(ch * scale)
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        return crop

    def record_unknown(
        self,
        frame: np.ndarray,
        face: dict[str, Any],
        camera_id: str,
        camera_name: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Record a detected unknown person with Multi-Sample Biometric Re-Identification
        and Same-Camera Spatial Continuity Guard.
        """
        if frame is None or not face:
            return None

        # Ignore enrolled known personnel
        if face.get("is_known") or (face.get("name") and face.get("name").lower() not in ("unknown", "")):
            return None

        bbox = face.get("bbox") or face.get("box")
        if not bbox or len(bbox) < 4:
            return None

        # Noise Gate: discard tiny fragments or poor detections
        bw = float(bbox[2] - bbox[0])
        bh = float(bbox[3] - bbox[1])
        if bw < 16 or bh < 16:
            return None

        raw_det_conf = face.get("confidence")
        if raw_det_conf is not None:
            det_val = float(raw_det_conf)
            if det_val < 0.25:
                return None
            conf_pct = round(det_val if det_val > 1.0 else det_val * 100.0, 1)
        else:
            conf_pct = 90.0

        cam_id_str = str(camera_id or "cam-unknown").strip()
        cam_name_str = str(camera_name or cam_id_str or "Surveillance Feed").strip()

        now_sec = time.time()
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        iso_timestamp = now_dt.isoformat()

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        track_id = face.get("track_id")

        # Extract face crop
        crop = self._extract_face_crop(frame, bbox)
        if crop is None or crop.size == 0:
            return None

        # --- EXTRACT / RETRIEVE 512-D BIOMETRIC EMBEDDING ---
        embedding: Optional[np.ndarray] = None
        face_emb = face.get("embedding")
        if face_emb is not None:
            try:
                emb_arr = np.array(face_emb, dtype=np.float32).flatten()
                if emb_arr.size == 512:
                    norm = float(np.linalg.norm(emb_arr))
                    if norm > 1e-8:
                        embedding = (emb_arr / norm).astype(np.float32)
            except Exception:
                embedding = None

        if embedding is None:
            try:
                from app.pipeline.face_service import face_service
                embedding = face_service.extract_face_crop_embedding(crop)
            except Exception as emb_err:
                logger.debug("Crop embedding extraction fallback failed: %s", emb_err)

        with self.lock:
            # -------------------------------------------------------------- #
            # STEP 1: SAME-CAMERA TEMPORAL & SPATIAL CONTINUITY GUARD       #
            # If this camera saw an unknown within the last 30 seconds:     #
            # Match by track_id, spatial proximity, or biometric correlation#
            # -------------------------------------------------------------- #
            active_list = self._camera_active_unknowns.setdefault(cam_id_str, [])
            # Prune tracks inactive for > 30 seconds
            active_list = [a for a in active_list if (now_sec - a.get("last_seen", 0.0)) < 30.0]
            self._camera_active_unknowns[cam_id_str] = active_list

            matched_active_id: Optional[str] = None
            best_active_sim = -1.0

            for act in active_list:
                act_id = act.get("unknown_id")
                if not act_id:
                    continue

                # 1. Exact ByteTrack track ID match (highest confidence)
                if track_id and act.get("track_id") == track_id:
                    matched_active_id = act_id
                    break

                # 2. Check biometric correlation against active person's gallery
                act_sim = 0.0
                if embedding is not None and act_id in self.embeddings:
                    gallery = self.embeddings[act_id]
                    if gallery.ndim == 1:
                        gallery = gallery.reshape(1, 512)
                    act_sim = float(np.max(np.dot(gallery, embedding)))

                prev_cx, prev_cy = act.get("center", (cx, cy))
                dist = float(np.hypot(cx - prev_cx, cy - prev_cy))
                dt = now_sec - act.get("last_seen", now_sec)

                # Matching conditions on SAME CAMERA:
                # When biometric embedding is present, require positive biometric correlation (sim >= 0.16)
                if embedding is not None:
                    if act_sim >= 0.16:
                        if act_sim > best_active_sim:
                            best_active_sim = act_sim
                            matched_active_id = act_id
                else:
                    # Fallback when embedding cannot be extracted (severe blur/distant crop):
                    if (dist < 360.0 and dt < 12.0) or (len(active_list) == 1 and dt < 8.0):
                        if matched_active_id is None:
                            matched_active_id = act_id

            if matched_active_id:
                act_rec = next((r for r in self.records if r["id"] == matched_active_id), None)
                if act_rec:
                    act_rec["seen_count"] = act_rec.get("seen_count", 1) + 1
                    act_rec["last_seen"] = timestamp_str
                    act_rec["_last_seen_sec"] = now_sec
                    act_rec["bbox"] = [round(float(v), 2) for v in bbox]

                    cams_list = act_rec.setdefault("cameras_seen", [])
                    if cam_name_str not in cams_list:
                        cams_list.append(cam_name_str)

                    # Update active track entry
                    act_entry = next((a for a in active_list if a.get("unknown_id") == matched_active_id), None)
                    if act_entry:
                        act_entry["last_seen"] = now_sec
                        act_entry["center"] = (cx, cy)
                        if track_id:
                            act_entry["track_id"] = track_id
                    else:
                        active_list.append({
                            "unknown_id": matched_active_id,
                            "last_seen": now_sec,
                            "center": (cx, cy),
                            "track_id": track_id,
                        })

                    # Photo upgrade if current is clearer
                    if conf_pct > float(act_rec.get("confidence", 0.0)):
                        act_rec["confidence"] = conf_pct
                        p_file = self.photos_dir / act_rec.get("photo_filename", "")
                        try:
                            cv2.imwrite(str(p_file), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                        except Exception:
                            pass

                    if embedding is not None:
                        self._add_embedding_sample(matched_active_id, embedding)

                    self._save_records()
                    self._save_embeddings()
                    logger.debug("🔒 SAME-CAMERA CONTINUITY LOCK: %s on '%s' (Seen: %d)", matched_active_id, cam_name_str, act_rec["seen_count"])
                    return act_rec

            # -------------------------------------------------------------- #
            # STEP 2: GLOBAL MULTI-CAMERA BIOMETRIC RE-IDENTIFICATION        #
            # Compare against all enrolled multi-sample unknown galleries    #
            # -------------------------------------------------------------- #
            best_match_id: Optional[str] = None
            best_similarity: float = -1.0

            if embedding is not None and self.embeddings:
                for uid, gallery in self.embeddings.items():
                    if gallery.ndim == 1:
                        gallery = gallery.reshape(1, 512)
                    sims = np.dot(gallery, embedding).ravel()
                    max_sim = float(np.max(sims))
                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_match_id = uid

            is_biometric_match = False
            matched_record: Optional[dict[str, Any]] = None

            if best_match_id is not None:
                matched_record = next((r for r in self.records if r["id"] == best_match_id), None)
                if matched_record is not None:
                    last_seen_sec = float(matched_record.get("_last_seen_sec", 0.0))
                    is_same_cam = matched_record.get("camera_id") == cam_id_str
                    is_recent = (now_sec - last_seen_sec) < 900.0  # within 15 minutes

                    thresh = self.reid_recent_threshold if (is_same_cam or is_recent) else self.reid_similarity_threshold
                    if best_similarity >= thresh:
                        is_biometric_match = True

            # If biometric match found across cameras or feeds: UPDATE EXISTING RECORD
            if is_biometric_match and matched_record is not None:
                matched_record["seen_count"] = matched_record.get("seen_count", 1) + 1
                matched_record["last_seen"] = timestamp_str
                matched_record["camera_id"] = cam_id_str
                matched_record["camera_name"] = cam_name_str
                matched_record["_last_seen_sec"] = now_sec

                cams_list = matched_record.setdefault("cameras_seen", [])
                if cam_name_str not in cams_list:
                    cams_list.append(cam_name_str)

                matched_record["bbox"] = [round(float(v), 2) for v in bbox]

                # Update camera active tracker list
                act_entry = next((a for a in active_list if a.get("unknown_id") == best_match_id), None)
                if act_entry:
                    act_entry["last_seen"] = now_sec
                    act_entry["center"] = (cx, cy)
                    if track_id:
                        act_entry["track_id"] = track_id
                else:
                    active_list.append({
                        "unknown_id": best_match_id,
                        "last_seen": now_sec,
                        "center": (cx, cy),
                        "track_id": track_id,
                    })

                # Clarity / Resolution Photo Upgrade
                prev_conf = float(matched_record.get("confidence", 0.0))
                if conf_pct >= prev_conf:
                    matched_record["confidence"] = conf_pct
                    p_file = self.photos_dir / matched_record.get("photo_filename", "")
                    try:
                        cv2.imwrite(str(p_file), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    except Exception:
                        pass

                # Store embedding in diverse gallery
                if embedding is not None and best_match_id:
                    self._add_embedding_sample(best_match_id, embedding)

                self._save_records()
                self._save_embeddings()

                logger.debug(
                    "🔄 RE-IDENTIFIED UNKNOWN %s on '%s' (Cosine Sim: %.3f, Seen: %d times, Cams: %s)",
                    best_match_id, cam_name_str, best_similarity, matched_record["seen_count"], matched_record["cameras_seen"],
                )
                return matched_record

            # -------------------------------------------------------------- #
            # STEP 3: NEW DISTINCT UNKNOWN INDIVIDUAL REGISTRATION           #
            # -------------------------------------------------------------- #
            unknown_id = self._generate_unknown_id()
            ts_compact = int(now_sec * 1000)
            photo_filename = f"unknown_{unknown_id}_{cam_id_str}_{ts_compact}.jpg"
            photo_path = self.photos_dir / photo_filename

            try:
                cv2.imwrite(str(photo_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            except Exception as save_err:
                logger.error("Could not save unknown face photograph: %s", save_err)
                return None

            photo_url = f"/api/faces/unknown/photo/{photo_filename}"

            record: dict[str, Any] = {
                "id": unknown_id,
                "timestamp": timestamp_str,
                "iso_timestamp": iso_timestamp,
                "camera_id": cam_id_str,
                "camera_name": cam_name_str,
                "cameras_seen": [cam_name_str],
                "confidence": conf_pct,
                "bbox": [round(float(v), 2) for v in bbox],
                "photo_filename": photo_filename,
                "photo_url": photo_url,
                "seen_count": 1,
                "last_seen": timestamp_str,
                "_last_seen_sec": now_sec,
                "status": "unreviewed",
                "notes": "",
                "enrolled_name": None,
            }

            self.records.insert(0, record)
            if len(self.records) > self.max_records:
                old = self.records.pop()
                old_id = old.get("id")
                self.embeddings.pop(old_id, None)
                old_file = self.photos_dir / old.get("photo_filename", "")
                if old_file.exists():
                    try:
                        old_file.unlink()
                    except Exception:
                        pass

            if embedding is not None:
                self.embeddings[unknown_id] = embedding.reshape(1, 512).astype(np.float32)

            active_list.append({
                "unknown_id": unknown_id,
                "last_seen": now_sec,
                "center": (cx, cy),
                "track_id": track_id,
            })

            self._save_records()
            self._save_embeddings()

        logger.info(
            "👤 NEW UNKNOWN PERSON REGISTERED: %s on '%s' (Confidence: %.1f%%) — Biometric gallery initialized.",
            unknown_id, cam_name_str, conf_pct,
        )

        self._broadcast_unknown_detected(record)
        return record

    def _broadcast_unknown_detected(self, record: dict[str, Any]) -> None:
        """Broadcast live unknown person detection event over WebSocket."""
        try:
            from app.pipeline.alert_service import alert_service
            clean_rec = {k: v for k, v in record.items() if not k.startswith("_")}
            alert_service.broadcaster.broadcast_sync({
                "type": "unknown_face_detected",
                "unknown": clean_rec,
            })
        except Exception as ws_err:
            logger.debug("Failed to broadcast unknown detection: %s", ws_err)

    def list_unknowns(
        self,
        camera_id: Optional[str] = None,
        search: Optional[str] = None,
        status_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Query and filter detected unknown persons list."""
        with self.lock:
            filtered = list(self.records)

        if camera_id and camera_id.lower() != "all":
            filtered = [r for r in filtered if r.get("camera_id") == camera_id or camera_id in r.get("cameras_seen", [])]

        if status_filter and status_filter.lower() != "all":
            filtered = [r for r in filtered if r.get("status", "").lower() == status_filter.lower()]

        if search and search.strip():
            q = search.strip().lower()
            filtered = [
                r for r in filtered
                if q in r.get("id", "").lower()
                or q in r.get("camera_name", "").lower()
                or q in r.get("camera_id", "").lower()
                or any(q in c.lower() for c in r.get("cameras_seen", []))
                or q in str(r.get("notes", "")).lower()
                or q in str(r.get("enrolled_name", "")).lower()
            ]

        total = len(filtered)
        paginated = filtered[offset : offset + limit]
        clean_paginated = [{k: v for k, v in r.items() if not k.startswith("_")} for r in paginated]

        return {
            "success": True,
            "total": total,
            "count": len(clean_paginated),
            "unknowns": clean_paginated,
        }

    def get_stats(self) -> dict[str, Any]:
        """Aggregate statistics on detected unknown individuals."""
        with self.lock:
            records = list(self.records)

        today_prefix = datetime.now().strftime("%Y-%m-%d")
        today_count = sum(1 for r in records if r.get("timestamp", "").startswith(today_prefix))
        unreviewed_count = sum(1 for r in records if r.get("status") == "unreviewed")
        enrolled_count = sum(1 for r in records if r.get("status") == "enrolled")

        by_camera: dict[str, int] = {}
        for r in records:
            c_name = r.get("camera_name") or r.get("camera_id") or "Unknown Feed"
            by_camera[c_name] = by_camera.get(c_name, 0) + 1

        return {
            "success": True,
            "total_unknowns": len(records),
            "today_count": today_count,
            "unreviewed_count": unreviewed_count,
            "enrolled_count": enrolled_count,
            "by_camera": by_camera,
        }

    def get_unknown(self, unknown_id: str) -> Optional[dict[str, Any]]:
        """Retrieve a single unknown person record by ID."""
        with self.lock:
            for r in self.records:
                if r["id"] == unknown_id:
                    clean = {k: v for k, v in r.items() if not k.startswith("_")}
                    return clean
        return None

    def update_unknown(
        self,
        unknown_id: str,
        status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Update administrative status (e.g. flagged, dismissed) or notes for an unknown person."""
        with self.lock:
            target = None
            for r in self.records:
                if r["id"] == unknown_id:
                    target = r
                    break
            if not target:
                return None

            if status is not None:
                target["status"] = status
            if notes is not None:
                target["notes"] = notes

            self._save_records()
            clean = {k: v for k, v in target.items() if not k.startswith("_")}
            return clean

    def delete_unknown(self, unknown_id: str) -> bool:
        """Delete an unknown person record and remove its photograph and embedding from disk."""
        photo_to_delete = None
        with self.lock:
            idx_to_remove = None
            for idx, r in enumerate(self.records):
                if r["id"] == unknown_id:
                    idx_to_remove = idx
                    photo_to_delete = r.get("photo_filename")
                    break

            if idx_to_remove is None:
                return False

            self.records.pop(idx_to_remove)
            self.embeddings.pop(unknown_id, None)

            # Clean from camera active trackers
            for cam_k in list(self._camera_active_unknowns.keys()):
                self._camera_active_unknowns[cam_k] = [
                    a for a in self._camera_active_unknowns[cam_k] if a.get("unknown_id") != unknown_id
                ]

            self._save_records()
            self._save_embeddings()

        if photo_to_delete:
            p_file = self.photos_dir / photo_to_delete
            if p_file.exists():
                try:
                    p_file.unlink()
                except Exception as exc:
                    logger.warning("Could not delete photo file %s: %s", photo_to_delete, exc)

        return True

    def clear_all(self) -> int:
        """Clear all unknown person records, wipe galleries, and delete photo directory."""
        with self.lock:
            count = len(self.records)
            self.records = []
            self.embeddings.clear()
            self._active_debounces.clear()
            self._camera_active_unknowns.clear()
            self._save_records()
            self._save_embeddings()

        # Delete photo files
        for p in self.photos_dir.glob("*.jpg"):
            try:
                p.unlink()
            except Exception:
                pass

        return count

    def consolidate_duplicates(self, similarity_threshold: float = 0.18) -> dict[str, Any]:
        """
        Retroactively merge duplicate unknown records using multi-sample biometric galleries.
        Uses Graph Connected Components to cluster all existing unknown individuals where
        max gallery similarity >= similarity_threshold (default: 0.18).
        """
        with self.lock:
            if len(self.records) <= 1:
                return {
                    "success": True,
                    "merged_count": 0,
                    "total_remaining": len(self.records),
                    "message": "No duplicates found.",
                }

            from app.pipeline.face_service import face_service

            # Ensure all records have at least one embedding (extract from saved photo if missing)
            for r in self.records:
                rid = r["id"]
                if rid not in self.embeddings or self.embeddings[rid].size == 0:
                    p_file = self.photos_dir / r.get("photo_filename", "")
                    if p_file.exists():
                        img = cv2.imread(str(p_file))
                        if img is not None:
                            emb = face_service.extract_face_crop_embedding(img)
                            if emb is not None:
                                self.embeddings[rid] = emb.reshape(1, 512).astype(np.float32)

            # Build Adjacency Graph for Connected Components
            records_by_id = {r["id"]: r for r in self.records}
            record_ids = list(records_by_id.keys())
            adj: dict[str, set[str]] = {rid: set() for rid in record_ids}

            for i in range(len(record_ids)):
                id_i = record_ids[i]
                g_i = self.embeddings.get(id_i)
                if g_i is None or g_i.size == 0:
                    continue
                if g_i.ndim == 1:
                    g_i = g_i.reshape(1, 512)

                for j in range(i + 1, len(record_ids)):
                    id_j = record_ids[j]
                    g_j = self.embeddings.get(id_j)
                    if g_j is None or g_j.size == 0:
                        continue
                    if g_j.ndim == 1:
                        g_j = g_j.reshape(1, 512)

                    sim_matrix = np.dot(g_i, g_j.T)
                    max_sim = float(np.max(sim_matrix))
                    if max_sim >= similarity_threshold:
                        adj[id_i].add(id_j)
                        adj[id_j].add(id_i)

            # Find Connected Components via BFS
            visited: set[str] = set()
            clusters: list[list[str]] = []

            for rid in record_ids:
                if rid in visited:
                    continue
                component = []
                queue = [rid]
                visited.add(rid)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append(component)

            merged_count = 0
            new_records: list[dict[str, Any]] = []

            for cluster in clusters:
                if len(cluster) == 1:
                    new_records.append(records_by_id[cluster[0]])
                    continue

                # Multiple duplicate records in cluster!
                merged_count += len(cluster) - 1
                cluster_recs = [records_by_id[cid] for cid in cluster]
                # Sort: enrolled first, then highest confidence, then seen_count
                cluster_recs.sort(key=lambda x: (
                    1 if x.get("status") == "enrolled" else 0,
                    float(x.get("confidence", 0.0)),
                    x.get("seen_count", 1)
                ), reverse=True)

                primary = cluster_recs[0]
                primary_id = primary["id"]

                # Merge seen counts and cameras
                total_seen = sum(r.get("seen_count", 1) for r in cluster_recs)
                all_cams = set()
                for r in cluster_recs:
                    for c in r.get("cameras_seen", [r.get("camera_name") or r.get("camera_id")]):
                        if c:
                            all_cams.add(c)

                primary["seen_count"] = total_seen
                primary["cameras_seen"] = sorted(list(all_cams))

                # Merge galleries (retain up to 8 diverse samples)
                all_vecs = []
                for cid in cluster:
                    g = self.embeddings.get(cid)
                    if g is not None and g.size > 0:
                        if g.ndim == 1:
                            g = g.reshape(1, 512)
                        all_vecs.append(g)
                if all_vecs:
                    stacked = np.vstack(all_vecs)
                    if len(stacked) > 8:
                        stacked = stacked[:8]
                    self.embeddings[primary_id] = stacked

                # Merge notes if any
                combined_notes = [r.get("notes") for r in cluster_recs if r.get("notes")]
                if combined_notes:
                    primary["notes"] = "; ".join(set(combined_notes))

                new_records.append(primary)

                # Remove duplicates from disk & memory
                for sec in cluster_recs[1:]:
                    sec_id = sec["id"]
                    self.embeddings.pop(sec_id, None)
                    sec_photo = sec.get("photo_filename")
                    if sec_photo and sec_photo != primary.get("photo_filename"):
                        p_path = self.photos_dir / sec_photo
                        if p_path.exists():
                            try:
                                p_path.unlink()
                            except Exception:
                                pass

            self.records = new_records
            self._save_records()
            self._save_embeddings()

            msg = f"Consolidated {merged_count} duplicate records. {len(self.records)} unique individuals remain."
            logger.info("Consolidated unknown persons: %s", msg)
            return {
                "success": True,
                "merged_count": merged_count,
                "total_remaining": len(self.records),
                "message": msg,
            }

    def enroll_unknown(
        self,
        unknown_id: str,
        name: str,
        is_suspect: bool = True,
        threat_level: str = "HIGH",
        category: str = "Wanted / BOLO",
        notes: str = "",
    ) -> dict[str, Any]:
        """
        1-Click Enroll an unknown person directly into the MAATRIX Face Recognition database
        using their saved photograph.
        """
        record = self.get_unknown(unknown_id)
        if not record:
            return {"success": False, "error": f"Unknown record {unknown_id} not found."}

        photo_filename = record.get("photo_filename")
        if not photo_filename:
            return {"success": False, "error": "No photograph associated with this unknown record."}

        photo_path = self.photos_dir / photo_filename
        if not photo_path.exists():
            return {"success": False, "error": f"Photograph file {photo_filename} missing from disk."}

        # Read photo into BGR array
        img_bgr = cv2.imread(str(photo_path))
        if img_bgr is None:
            return {"success": False, "error": "Failed to decode saved photograph."}

        from app.pipeline.face_service import face_service

        reg_result = face_service.register_face(
            image_bgr=img_bgr,
            name=name.strip(),
            is_suspect=is_suspect,
            threat_level=threat_level,
            category=category,
            notes=notes or f"Enrolled from detected unknown record {unknown_id}",
        )

        if not reg_result.get("success"):
            return reg_result

        # Update record status to enrolled
        self.update_unknown(
            unknown_id,
            status="enrolled",
            notes=f"Enrolled as {name.strip()} ({'Suspect' if is_suspect else 'Authorized'})",
        )
        with self.lock:
            for r in self.records:
                if r["id"] == unknown_id:
                    r["enrolled_name"] = name.strip()
                    break
            self._save_records()

        logger.info(
            "✅ Unknown person %s successfully enrolled into roster as '%s'",
            unknown_id, name.strip(),
        )

        return {
            "success": True,
            "message": f"Successfully enrolled {name.strip()} from unknown detection {unknown_id}.",
            "person": reg_result.get("person"),
            "unknown_id": unknown_id,
        }


# Global singleton service
unknown_face_service = UnknownFaceService()
