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
from app.tracking.topology import camera_topology, CameraTopology
from app.tracking.calibration_service import calibration_service

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
        footprint_ground_m: Optional[tuple[float, float]] = None,
        footprint_gps: Optional[tuple[float, float]] = None,
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
                "spatial_score": 1.0,
                "fused_score": 1.0,
                "speed_kmh": 0.0,
                "footprint_ground_m": [round(float(v), 2) for v in footprint_ground_m] if footprint_ground_m else None,
                "footprint_gps": [round(float(v), 6) for v in footprint_gps] if footprint_gps else None,
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
        speed_kmh: Optional[float] = None,
        spatial_score: Optional[float] = None,
        fused_score: Optional[float] = None,
        footprint_ground_m: Optional[tuple[float, float]] = None,
        footprint_gps: Optional[tuple[float, float]] = None,
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
                "spatial_score": round(float(spatial_score), 3) if spatial_score is not None else 1.0,
                "fused_score": round(float(fused_score), 3) if fused_score is not None else round(float(reid_score), 3),
                "speed_kmh": round(float(speed_kmh), 2) if speed_kmh is not None else None,
                "footprint_ground_m": [round(float(v), 2) for v in footprint_ground_m] if footprint_ground_m else None,
                "footprint_gps": [round(float(v), 6) for v in footprint_gps] if footprint_gps else None,
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
        recorded_speeds = [wp["speed_kmh"] for wp in self.waypoints if wp.get("speed_kmh") is not None and wp.get("speed_kmh", 0) > 0]
        avg_speed = round(float(sum(recorded_speeds) / len(recorded_speeds)), 2) if recorded_speeds else 0.0

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
            "average_speed_kmh": avg_speed,
        }


class GlobalTraceManager:
    """
    Central Thread-Safe Manager for Cross-Camera Multi-Target Multi-Camera Tracking (MTMCT).
    Bridges local single-camera ByteTrack IDs into unified Global Trace IDs using
    Deep Appearance Re-ID (OSNet) coupled with Spatio-Temporal & Kinematic Gating.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        retention_ttl_seconds: float = 600.0,
        max_pedestrian_speed_kmh: float = 25.0,
        max_vehicle_speed_kmh: float = 100.0,
        min_transit_time_seconds: float = 0.5,
        spatial_collocation_threshold_m: float = 15.0,
        reid_weight: float = 0.70,
        spatial_weight: float = 0.30,
        topology: Optional[CameraTopology] = None,
    ) -> None:
        self._traces: dict[str, GlobalTrace] = {}
        # Maps (camera_id, local_track_id) -> global_trace_id
        self._camera_local_map: dict[tuple[str, int], str] = {}
        # Maps (camera_id, local_track_id) -> last timestamp seen
        self._local_last_seen: dict[tuple[str, int], float] = {}

        self.next_trace_num: int = 1
        self.similarity_threshold: float = similarity_threshold
        self.retention_ttl_seconds: float = retention_ttl_seconds
        self.max_pedestrian_speed_kmh: float = max_pedestrian_speed_kmh
        self.max_vehicle_speed_kmh: float = max_vehicle_speed_kmh
        self.min_transit_time_seconds: float = min_transit_time_seconds
        self.spatial_collocation_threshold_m: float = spatial_collocation_threshold_m

        # Multi-factor weights: appearance vs spatial topology
        self.reid_weight: float = reid_weight
        self.spatial_weight: float = spatial_weight
        self.topology: CameraTopology = topology or camera_topology
        self.lock = threading.Lock()

        # Cache camera metadata and GPS coordinates (lat, lon)
        self._camera_coords: dict[str, tuple[float, float]] = {}
        self._camera_meta: dict[str, dict[str, Any]] = {}
        self._load_cameras_cache()

    def _load_cameras_cache(self) -> None:
        """Cache camera GPS locations and metadata from cameras.json."""
        try:
            if DATA_FILE.exists():
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    cams = json.load(f)
                    for c in cams:
                        cid = str(c.get("id"))
                        self._camera_meta[cid] = c
                        lat = c.get("latitude")
                        lon = c.get("longitude")
                        if lat is not None and lon is not None:
                            try:
                                self._camera_coords[cid] = (float(lat), float(lon))
                            except (ValueError, TypeError):
                                pass
        except Exception as err:
            logger.debug("Could not load camera coordinates cache: %s", err)

    def register_camera_coordinates(self, camera_id: str, latitude: float, longitude: float) -> None:
        """Register or update GPS coordinates for a specific camera ID."""
        with self.lock:
            self._camera_coords[str(camera_id)] = (float(latitude), float(longitude))

    def get_camera_coordinates(self, camera_id: str) -> Optional[tuple[float, float]]:
        """Retrieve cached GPS coordinates (lat, lon) for a camera."""
        return self._camera_coords.get(str(camera_id))

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
        timestamp: Optional[float] = None,
        is_vehicle: bool = False,
    ) -> tuple[str, bool, float]:
        """
        Assign or update a unified Global Trace ID for a detected person or vehicle.

        Returns:
            tuple of (trace_id, is_cross_camera_match, fused_confidence_score)
        """
        now = float(timestamp) if timestamp is not None else time.time()
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

            # Case 2: New local track in this camera -> Perform Cross-Camera Multi-Factor Re-ID
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

            # Project 2D bounding box footpoint onto ground plane and geodetic GPS
            footprint_ground_m, footprint_gps = calibration_service.project_box_footprint(
                camera_id=camera_id,
                box=box,
                frame_resolution=(w, h),
            )

            # Search existing active traces for cross-camera appearance match
            best_match: Optional[GlobalTrace] = None
            best_score: float = -1.0
            best_reid_score: float = 0.0
            best_spatial_score: float = 0.0
            best_speed_kmh: Optional[float] = None
            best_dt_sec: float = 0.0

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

                # --- Kinematic & Spatio-Temporal Gating ---
                cand_speed_kmh: Optional[float] = None
                dist_km: Optional[float] = None
                dt_sec = max(0.001, now - candidate.last_seen)

                if candidate.last_camera_id != camera_id:
                    coord_curr = self.get_camera_coordinates(camera_id)
                    coord_prev = self.get_camera_coordinates(candidate.last_camera_id)

                    if coord_curr is not None and coord_prev is not None:
                        dist_km = calc_haversine_km(coord_prev[0], coord_prev[1], coord_curr[0], coord_curr[1])
                        dist_m = dist_km * 1000.0

                        # If cameras are geographically separated beyond collocation threshold
                        if dist_m > self.spatial_collocation_threshold_m:
                            # 1. Enforce minimum physical transit time
                            if dt_sec < self.min_transit_time_seconds:
                                logger.debug(
                                    "Kinematic gate rejected %s: dt=%.2fs < min_transit=%.2fs between %s and %s",
                                    candidate.trace_id, dt_sec, self.min_transit_time_seconds, candidate.last_camera_id, camera_id
                                )
                                continue

                            # 2. Enforce maximum physical velocity (teleportation rejection)
                            cand_speed_kmh = dist_km / (dt_sec / 3600.0)
                            speed_limit = self.max_vehicle_speed_kmh if is_vehicle else self.max_pedestrian_speed_kmh
                            if cand_speed_kmh > speed_limit:
                                logger.info(
                                    "🚨 TELEPORTATION REJECTED: Candidate %s requires %.1f km/h between %s and %s (dist=%.2f km, dt=%.1f s). Limit is %.1f km/h.",
                                    candidate.trace_id, cand_speed_kmh, candidate.last_camera_id, camera_id, dist_km, dt_sec, speed_limit
                                )
                                continue

                # 1. Multi-shot visual Re-ID appearance score (S_reid)
                cand_features = [candidate.feature] + candidate.feature_history
                scores = [reid_service.compute_similarity(feature, f) for f in cand_features]
                cand_reid_score = max(scores) if scores else -1.0

                if cand_reid_score < 0.40:
                    # Clear visual mismatch: skip computing topology affinity
                    continue

                # 2. Spatio-temporal affinity from Camera Topology (S_spatial)
                if candidate.last_camera_id != camera_id:
                    cand_spatial_score = self.topology.compute_spatial_affinity(
                        from_camera=candidate.last_camera_id,
                        to_camera=camera_id,
                        dt_sec=dt_sec,
                        dist_km=dist_km,
                    )
                else:
                    cand_spatial_score = 1.0

                # 3. Fused Multi-Factor Score:
                cand_fused_score = (
                    self.reid_weight * cand_reid_score + self.spatial_weight * cand_spatial_score
                )

                if cand_fused_score > best_score:
                    best_score = cand_fused_score
                    best_reid_score = cand_reid_score
                    best_spatial_score = cand_spatial_score
                    best_match = candidate
                    best_speed_kmh = cand_speed_kmh
                    best_dt_sec = dt_sec if candidate.last_camera_id != camera_id else 0.0

            # Decision Boundary:
            if best_match is not None and best_score >= self.similarity_threshold:
                # Cross-Camera Match Found!
                trace = best_match
                trace.last_seen = now
                trace.total_sightings += 1
                trace.update_feature(feature)

                # Record new camera waypoint transition
                if trace.last_camera_id != camera_id:
                    # Online Learning: Record transition in topology graph
                    if best_dt_sec > 0:
                        self.topology.record_transition(trace.last_camera_id, camera_id, best_dt_sec)

                    trace.add_waypoint(
                        camera_id=camera_id,
                        camera_name=camera_name,
                        box=box,
                        confidence=confidence,
                        reid_score=best_reid_score,
                        timestamp=now,
                        speed_kmh=best_speed_kmh,
                        spatial_score=best_spatial_score,
                        fused_score=best_score,
                        footprint_ground_m=footprint_ground_m,
                        footprint_gps=footprint_gps,
                    )
                    logger.info(
                        "🚨 CROSS-CAMERA RE-ID SUCCESS: %s transitioned from %s to %s (Fused=%.3f, ReID=%.3f, Spatial=%.3f, Speed=%.1f km/h, Local ID #%d)",
                        trace.trace_id,
                        trace.waypoints[-2]["camera_name"] if len(trace.waypoints) > 1 else "Unknown",
                        camera_name,
                        best_score,
                        best_reid_score,
                        best_spatial_score,
                        best_speed_kmh or 0.0,
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
                footprint_ground_m=footprint_ground_m,
                footprint_gps=footprint_gps,
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

        cameras_map: dict[str, dict[str, Any]] = self._camera_meta
        if not cameras_map and DATA_FILE.exists():
            self._load_cameras_cache()
            cameras_map = self._camera_meta

        enriched_waypoints = []
        total_distance = 0.0
        prev_coords: Optional[tuple[float, float]] = None
        prev_time = 0.0

        for idx, wp in enumerate(waypoints_copy):
            cam_id = str(wp.get("camera_id", ""))
            cam_info = cameras_map.get(cam_id, {})
            footprint_gps = wp.get("footprint_gps")
            if footprint_gps and len(footprint_gps) == 2 and footprint_gps[0] is not None:
                lat = float(footprint_gps[0])
                lng = float(footprint_gps[1])
            else:
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
                    "spatial_score": wp.get("spatial_score", 1.0),
                    "fused_score": wp.get("fused_score", wp.get("reid_score", 1.0)),
                    "confidence": wp.get("confidence", 0.0),
                    "speed_kmh": wp.get("speed_kmh", 0.0),
                    "footprint_ground_m": wp.get("footprint_ground_m"),
                    "footprint_gps": wp.get("footprint_gps"),
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
            if not self._camera_coords:
                self._load_cameras_cache()
            logger.info("GlobalTraceManager reset.")


global_trace_manager = GlobalTraceManager()
