from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from app.pipeline.reid_service import reid_service

logger = logging.getLogger("SIH26187.GlobalTracker")

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "cameras.json"


def calc_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance in kilometers between two GPS coordinates."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(r * c, 3)


class GlobalTrace:
    """
    Persistent global identity representing a person observed across one or more cameras.
    """

    def __init__(
        self,
        trace_id: str,
        feature: np.ndarray,
        camera_id: str,
        camera_name: str,
        box: list[float],
        confidence: float,
        timestamp: float,
        suspect_name: Optional[str] = None,
        threat_level: Optional[str] = None,
        category: Optional[str] = None,
    ) -> None:
        self.trace_id: str = trace_id
        self.feature: np.ndarray = feature
        self.feature_history: list[np.ndarray] = [feature]
        self.first_seen: float = timestamp
        self.last_seen: float = timestamp
        self.last_camera_id: str = camera_id
        self.last_camera_name: str = camera_name
        self.total_sightings: int = 1

        self.suspect_name: Optional[str] = suspect_name
        self.threat_level: Optional[str] = threat_level
        self.category: Optional[str] = category
        self.is_suspect: bool = bool(suspect_name and (threat_level or "ALERT") != "LOW")

        # Chronological list of camera sightings / transit events
        self.waypoints: list[dict[str, Any]] = [
            {
                "step": 1,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "timestamp": timestamp,
                "box": [round(float(v), 2) for v in box],
                "confidence": round(float(confidence), 3),
                "reid_score": 1.0,
            }
        ]

    @property
    def is_cross_camera(self) -> bool:
        """Returns True if this person has been observed across multiple distinct cameras."""
        cams = {wp["camera_id"] for wp in self.waypoints}
        return len(cams) > 1

    def add_waypoint(
        self,
        camera_id: str,
        camera_name: str,
        box: list[float],
        confidence: float,
        reid_score: float,
        timestamp: float,
    ) -> None:
        self.waypoints.append(
            {
                "step": len(self.waypoints) + 1,
                "camera_id": camera_id,
                "camera_name": camera_name,
                "timestamp": timestamp,
                "box": [round(float(v), 2) for v in box],
                "confidence": round(float(confidence), 3),
                "reid_score": round(float(reid_score), 3),
            }
        )
        self.last_camera_id = camera_id
        self.last_camera_name = camera_name

    def update_feature(self, new_feature: np.ndarray, alpha: float = 0.85) -> None:
        """
        Update the aggregated appearance feature using Exponential Moving Average (EMA).
        Maintains an appearance history of up to 5 diverse captures.
        """
        if new_feature is None:
            return
        # EMA update
        updated = alpha * self.feature + (1.0 - alpha) * new_feature
        norm = float(np.linalg.norm(updated))
        if norm > 1e-6:
            self.feature = (updated / norm).astype(np.float32)

        # Store in history if sufficiently distinct from existing angles
        is_distinct = True
        for h in self.feature_history:
            if float(np.dot(h, new_feature)) > 0.92:
                is_distinct = False
                break
        if is_distinct:
            self.feature_history.append(new_feature)
            if len(self.feature_history) > 5:
                self.feature_history.pop(0)

    def to_dict(self) -> dict[str, Any]:
        """Convert GlobalTrace into JSON-serializable representation."""
        duration_sec = round(self.last_seen - self.first_seen, 1)
        unique_cameras = list(dict.fromkeys(wp["camera_id"] for wp in self.waypoints))

        return {
            "trace_id": self.trace_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_sec": duration_sec,
            "last_camera_id": self.last_camera_id,
            "last_camera_name": self.last_camera_name,
            "total_sightings": self.total_sightings,
            "unique_cameras_count": len(unique_cameras),
            "is_cross_camera": self.is_cross_camera,
            "is_suspect": self.is_suspect,
            "suspect_name": self.suspect_name,
            "threat_level": self.threat_level,
            "category": self.category,
            "waypoints_count": len(self.waypoints),
        }


class GlobalTraceManager:
    """
    Central Thread-Safe Manager for Cross-Camera Multi-Target Multi-Camera Tracking (MTMCT).
    Bridges local single-camera ByteTrack IDs into unified Global Trace IDs using
    Deep Appearance Re-ID (OSNet).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        retention_ttl_seconds: float = 600.0,
    ) -> None:
        self._traces: dict[str, GlobalTrace] = {}
        # Maps (camera_id, local_track_id) -> global_trace_id
        self._camera_local_map: dict[tuple[str, int], str] = {}
        # Maps (camera_id, local_track_id) -> last timestamp seen
        self._local_last_seen: dict[tuple[str, int], float] = {}

        self.next_trace_num: int = 1
        self.similarity_threshold: float = similarity_threshold
        self.retention_ttl_seconds: float = retention_ttl_seconds
        self.lock = threading.Lock()

    def update_track(
        self,
        camera_id: str,
        camera_name: str,
        local_track_id: int,
        box: list[float],
        frame_bgr: np.ndarray,
        confidence: float,
        suspect_name: Optional[str] = None,
        threat_level: Optional[str] = None,
        category: Optional[str] = None,
    ) -> tuple[str, bool, float]:
        """
        Assign or update a unified Global Trace ID for a detected person.

        Returns:
            tuple of (trace_id, is_cross_camera_match, reid_confidence_score)
        """
        now = time.time()
        map_key = (camera_id, int(local_track_id))

        with self.lock:
            self._prune_inactive_locked(now)

            # Case 1: Local track is ALREADY mapped to an existing GlobalTrace
            if map_key in self._camera_local_map:
                trace_id = self._camera_local_map[map_key]
                trace = self._traces.get(trace_id)
                if trace:
                    trace.last_seen = now
                    trace.total_sightings += 1
                    self._local_last_seen[map_key] = now

                    # Attach suspect identity if newly discovered
                    if suspect_name and not trace.suspect_name:
                        trace.suspect_name = suspect_name
                        trace.threat_level = threat_level or "HIGH"
                        trace.category = category or "Wanted / BOLO"
                        trace.is_suspect = True

                    return trace.trace_id, False, 1.0

            # Case 2: New local track in this camera -> Perform Cross-Camera Re-ID
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            crop = frame_bgr[y1:y2, x1:x2]
            feature = reid_service.extract_crop(crop)

            if feature is None:
                # Crop too small or degraded: assign fallback temporary ID
                trace_id = f"TRC-TEMP-{int(now) % 10000}"
                return trace_id, False, 0.0

            # Search existing active traces for cross-camera appearance match
            best_match: Optional[GlobalTrace] = None
            best_score: float = -1.0

            for candidate in self._traces.values():
                # Spatial Exclusivity Filter:
                # If candidate is currently active on the SAME camera right now under another local track ID,
                # they cannot be the exact same person.
                is_currently_on_same_camera = False
                for (other_cam, other_loc_id), other_tid in self._camera_local_map.items():
                    if other_tid == candidate.trace_id and other_cam == camera_id:
                        other_seen = self._local_last_seen.get((other_cam, other_loc_id), 0.0)
                        if (now - other_seen) < 2.0:  # Seen within last 2 seconds on this camera
                            is_currently_on_same_camera = True
                            break

                if is_currently_on_same_camera:
                    continue

                # Multi-shot feature comparison
                cand_features = [candidate.feature] + candidate.feature_history
                scores = [reid_service.compute_similarity(feature, f) for f in cand_features]
                cand_max_score = max(scores) if scores else -1.0

                if cand_max_score > best_score:
                    best_score = cand_max_score
                    best_match = candidate

            # Decision Boundary:
            if best_match is not None and best_score >= self.similarity_threshold:
                # Cross-Camera Match Found!
                trace = best_match
                trace.last_seen = now
                trace.total_sightings += 1
                trace.update_feature(feature)

                # Record new camera waypoint transition
                if trace.last_camera_id != camera_id:
                    trace.add_waypoint(
                        camera_id=camera_id,
                        camera_name=camera_name,
                        box=box,
                        confidence=confidence,
                        reid_score=best_score,
                        timestamp=now,
                    )
                    logger.info(
                        "🚨 CROSS-CAMERA RE-ID SUCCESS: %s transitioned from %s to %s (Score=%.3f, Local ID #%d)",
                        trace.trace_id,
                        trace.waypoints[-2]["camera_name"] if len(trace.waypoints) > 1 else "Unknown",
                        camera_name,
                        best_score,
                        local_track_id,
                    )

                # Attach suspect if present
                if suspect_name and not trace.suspect_name:
                    trace.suspect_name = suspect_name
                    trace.threat_level = threat_level or "HIGH"
                    trace.category = category or "Wanted / BOLO"
                    trace.is_suspect = True

                self._camera_local_map[map_key] = trace.trace_id
                self._local_last_seen[map_key] = now
                return trace.trace_id, True, best_score

            # No match found -> Initialize a brand new Global Trace
            trace_id = f"TRC-{self.next_trace_num:04d}"
            self.next_trace_num += 1

            new_trace = GlobalTrace(
                trace_id=trace_id,
                feature=feature,
                camera_id=camera_id,
                camera_name=camera_name,
                box=box,
                confidence=confidence,
                timestamp=now,
                suspect_name=suspect_name,
                threat_level=threat_level,
                category=category,
            )

            self._traces[trace_id] = new_trace
            self._camera_local_map[map_key] = trace_id
            self._local_last_seen[map_key] = now
            logger.info("New Global Trace initialized: %s on %s (Local ID #%d)", trace_id, camera_name, local_track_id)
            return trace_id, False, 1.0

    def get_trace(self, trace_id: str) -> Optional[GlobalTrace]:
        with self.lock:
            return self._traces.get(trace_id)

    def list_active_traces(self) -> list[dict[str, Any]]:
        with self.lock:
            traces_list = [t.to_dict() for t in self._traces.values()]
            # Sort with cross-camera and suspects at the top, then recent sightings
            traces_list.sort(key=lambda x: (x["is_cross_camera"], x["is_suspect"], x["last_seen"]), reverse=True)
            return traces_list

    def get_trace_trajectory(self, trace_id: str) -> dict[str, Any]:
        """
        Reconstruct the chronological multi-camera GPS trajectory for a given Global Trace ID.
        Calculates cumulative distance (km), elapsed transit times, and waypoint coordinates.
        """
        with self.lock:
            trace = self._traces.get(trace_id)
            if not trace:
                return {
                    "trace_id": trace_id,
                    "found": False,
                    "total_sightings": 0,
                    "waypoints": [],
                    "total_distance_km": 0.0,
                }
            waypoints_copy = list(trace.waypoints)
            suspect_name = trace.suspect_name
            threat_level = trace.threat_level
            category = trace.category

        cameras_map: dict[str, dict[str, Any]] = {}
        try:
            if DATA_FILE.exists():
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    cams = json.load(f)
                    for c in cams:
                        cameras_map[str(c.get("id"))] = c
        except Exception as err:
            logger.debug("Could not read cameras.json for global trace trajectory: %s", err)

        enriched_waypoints = []
        total_distance = 0.0
        prev_coords: Optional[tuple[float, float]] = None
        prev_time = 0.0

        for idx, wp in enumerate(waypoints_copy):
            cam_id = str(wp.get("camera_id", ""))
            cam_info = cameras_map.get(cam_id, {})
            lat = float(cam_info.get("latitude") or 32.7240 + (idx * 0.008))
            lng = float(cam_info.get("longitude") or 74.6710 + (idx * 0.008))

            delta_km = 0.0
            elapsed_min = 0.0
            wp_time = float(wp.get("timestamp", 0.0))

            if prev_coords is not None:
                delta_km = calc_haversine_km(prev_coords[0], prev_coords[1], lat, lng)
                total_distance += delta_km
                if prev_time > 0:
                    elapsed_min = round((wp_time - prev_time) / 60.0, 1)

            prev_coords = (lat, lng)
            prev_time = wp_time

            enriched_waypoints.append(
                {
                    "step": idx + 1,
                    "camera_id": cam_id,
                    "camera_name": wp.get("camera_name") or cam_info.get("name", f"Camera {cam_id}"),
                    "sector": cam_info.get("sector", "Sector A"),
                    "latitude": lat,
                    "longitude": lng,
                    "timestamp": wp_time,
                    "time_str": time.strftime("%H:%M:%S", time.localtime(wp_time)),
                    "delta_distance_km": round(delta_km, 3),
                    "elapsed_time_minutes": elapsed_min,
                    "reid_score": wp.get("reid_score", 1.0),
                    "confidence": wp.get("confidence", 0.0),
                }
            )

        last_wp = enriched_waypoints[-1] if enriched_waypoints else None

        return {
            "trace_id": trace_id,
            "found": True,
            "suspect_name": suspect_name,
            "threat_level": threat_level,
            "category": category,
            "total_sightings": len(enriched_waypoints),
            "total_distance_km": round(total_distance, 2),
            "last_location": last_wp["camera_name"] if last_wp else "Unknown",
            "last_seen_time": last_wp["time_str"] if last_wp else "",
            "waypoints": enriched_waypoints,
            "is_cross_camera": len({wp["camera_id"] for wp in enriched_waypoints}) > 1,
        }

    def _prune_inactive_locked(self, now: float) -> None:
        """Internal helper: remove traces inactive for longer than TTL."""
        expired_trace_ids = [
            tid for tid, t in self._traces.items() if (now - t.last_seen) > self.retention_ttl_seconds
        ]
        for tid in expired_trace_ids:
            del self._traces[tid]

        # Clean local map references
        expired_keys = [
            k for k, tid in self._camera_local_map.items() if tid in expired_trace_ids or (now - self._local_last_seen.get(k, 0.0)) > self.retention_ttl_seconds
        ]
        for k in expired_keys:
            self._camera_local_map.pop(k, None)
            self._local_last_seen.pop(k, None)

    def prune_inactive(self) -> int:
        with self.lock:
            now = time.time()
            before = len(self._traces)
            self._prune_inactive_locked(now)
            return before - len(self._traces)

    def reset(self) -> None:
        with self.lock:
            self._traces.clear()
            self._camera_local_map.clear()
            self._local_last_seen.clear()
            self.next_trace_num = 1
            logger.info("GlobalTraceManager reset.")


global_trace_manager = GlobalTraceManager()
