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
    Centralized Unknown Person Detection & Archival Service for MayaJaal.

    When an un-enrolled / unrecognized person is detected on any camera feed:
      1. Generates a distinct human-readable Unknown ID (e.g., UNK-7E2A).
      2. Extracts a high-clarity face portrait photograph with aspect-ratio padding.
      3. Saves the photograph persistently to disk with unique naming.
      4. Debounces rapid-fire detections per camera track so the database is not flooded.
      5. Records metadata (ID, real-time timestamp, camera name/ID, detection confidence, bbox).
      6. Dispatches real-time WebSocket notifications to the admin operational console.
      7. Allows admins to review, filter, delete, or 1-click enroll the unknown person into the roster.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data_dir = settings.model_directory.parent / "app" / "data" / "faces"
        self.unknowns_dir = self.data_dir / "unknowns"
        self.photos_dir = self.unknowns_dir / "photos"
        self.db_file = self.unknowns_dir / "unknown_records.json"

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.unknowns_dir.mkdir(parents=True, exist_ok=True)
        self.photos_dir.mkdir(parents=True, exist_ok=True)

        self.records: list[dict[str, Any]] = []
        self.max_records: int = 500

        # In-memory tracking cache for debouncing rapid consecutive frames:
        # Key: (camera_id, track_id) or (camera_id, spatial_bucket) -> { "unknown_id": str, "last_seen": float, "bbox": list }
        self._active_debounces: dict[str, dict[str, Any]] = {}
        self.debounce_cooldown: float = 12.0  # seconds between new distinct records for same physical track

        self._load_records()

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

    def _generate_unknown_id(self) -> str:
        """Generate a clean, memorable Unknown Person ID, e.g., UNK-9E3A."""
        short_hex = uuid.uuid4().hex[:4].upper()
        return f"UNK-{short_hex}"

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
        Record a detected unknown person.
        Extracts face photo crop, generates ID, debounces continuous frames,
        saves to disk, and notifies real-time listeners.
        """
        if frame is None or not face:
            return None

        # Verify this is indeed an unknown face
        if face.get("is_known") or (face.get("name") and face.get("name").lower() != "unknown"):
            return None

        bbox = face.get("bbox") or face.get("box")
        if not bbox or len(bbox) < 4:
            return None

        cam_id_str = str(camera_id or "cam-unknown").strip()
        cam_name_str = str(camera_name or cam_id_str or "Surveillance Feed").strip()

        now_sec = time.time()
        now_dt = datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        iso_timestamp = now_dt.isoformat()

        track_id = face.get("track_id")
        det_conf = float(face.get("calibrated_conf") or face.get("confidence", 0.85))
        if det_conf > 1.0 and det_conf <= 100.0:
            conf_pct = round(det_conf, 1)
        else:
            conf_pct = round(det_conf * 100.0, 1)

        # ------------------------------------------------------------------ #
        # DEBOUNCING: Prevent saving 30 photos/sec for same person in view    #
        # ------------------------------------------------------------------ #
        debounce_key = f"{cam_id_str}_track_{track_id}" if track_id else None

        # If no track_id, use spatial proximity bucket (center coordinates / 100)
        if not debounce_key:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            spatial_bucket = f"{int(cx // 80)}_{int(cy // 80)}"
            debounce_key = f"{cam_id_str}_spatial_{spatial_bucket}"

        with self.lock:
            # Clean up stale debounces (> 30s)
            stale_keys = [k for k, v in self._active_debounces.items() if (now_sec - v["last_seen"]) > 30.0]
            for k in stale_keys:
                del self._active_debounces[k]

            cached = self._active_debounces.get(debounce_key)
            if cached and (now_sec - cached["last_seen"]) < self.debounce_cooldown:
                # Update existing record's last seen & detection count
                cached["last_seen"] = now_sec
                rec_id = cached["unknown_id"]
                for r in self.records:
                    if r["id"] == rec_id:
                        r["seen_count"] = r.get("seen_count", 1) + 1
                        r["last_seen"] = timestamp_str
                        # If current detection is higher confidence, update confidence & bbox
                        if conf_pct > r.get("confidence", 0):
                            r["confidence"] = conf_pct
                            r["bbox"] = [round(float(v), 2) for v in bbox]
                        break
                return cached.get("record")

            # Extract face photograph crop
            crop = self._extract_face_crop(frame, bbox)
            if crop is None or crop.size == 0:
                return None

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
                "confidence": conf_pct,
                "bbox": [round(float(v), 2) for v in bbox],
                "photo_filename": photo_filename,
                "photo_url": photo_url,
                "seen_count": 1,
                "last_seen": timestamp_str,
                "status": "unreviewed",
                "notes": "",
                "enrolled_name": None,
            }

            self.records.insert(0, record)
            if len(self.records) > self.max_records:
                # Remove oldest record and optionally its photo
                old = self.records.pop()
                old_file = self.photos_dir / old.get("photo_filename", "")
                if old_file.exists():
                    try:
                        old_file.unlink()
                    except Exception:
                        pass

            self._save_records()

            self._active_debounces[debounce_key] = {
                "unknown_id": unknown_id,
                "last_seen": now_sec,
                "record": record,
            }

        logger.info(
            "👤 NEW UNKNOWN PERSON DETECTED: %s on '%s' (Confidence: %.1f%%) — Photograph saved as %s",
            unknown_id, cam_name_str, conf_pct, photo_filename,
        )

        # Broadcast live event to connected admin clients
        self._broadcast_unknown_detected(record)

        return record

    def _broadcast_unknown_detected(self, record: dict[str, Any]) -> None:
        """Broadcast live unknown person detection event over WebSocket."""
        try:
            from app.pipeline.alert_service import alert_service
            alert_service.broadcaster.broadcast_sync({
                "type": "unknown_face_detected",
                "unknown": record,
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
            filtered = [r for r in filtered if r.get("camera_id") == camera_id]

        if status_filter and status_filter.lower() != "all":
            filtered = [r for r in filtered if r.get("status", "").lower() == status_filter.lower()]

        if search and search.strip():
            q = search.strip().lower()
            filtered = [
                r for r in filtered
                if q in r.get("id", "").lower()
                or q in r.get("camera_name", "").lower()
                or q in r.get("camera_id", "").lower()
                or q in r.get("notes", "").lower()
                or q in r.get("enrolled_name", "").lower()
            ]

        total = len(filtered)
        paginated = filtered[offset : offset + limit]

        return {
            "success": True,
            "total": total,
            "count": len(paginated),
            "unknowns": paginated,
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
                    return dict(r)
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
            return dict(target)

    def delete_unknown(self, unknown_id: str) -> bool:
        """Delete an unknown person record and remove its photograph from disk."""
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
            self._save_records()

        if photo_to_delete:
            p_file = self.photos_dir / photo_to_delete
            if p_file.exists():
                try:
                    p_file.unlink()
                except Exception as exc:
                    logger.warning("Could not delete photo file %s: %s", photo_to_delete, exc)

        return True

    def clear_all(self) -> int:
        """Clear all unknown person records and wipe photos directory."""
        with self.lock:
            count = len(self.records)
            self.records = []
            self._active_debounces.clear()
            self._save_records()

        # Delete photo files
        for p in self.photos_dir.glob("*.jpg"):
            try:
                p.unlink()
            except Exception:
                pass

        return count

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
