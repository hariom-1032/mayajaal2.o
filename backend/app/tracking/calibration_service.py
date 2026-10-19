from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("SIH26187.CalibrationService")

DEFAULT_CALIBRATIONS_FILE = Path(__file__).resolve().parent.parent / "data" / "calibrations.json"


class CameraCalibration:
    """
    Homography and geometric calibration container for a single camera.
    Maps 2D image coordinates (pixels) <-> Real-world ground plane coordinates (meters)
    and geodetic GPS coordinates (lat, lon).
    """

    def __init__(
        self,
        camera_id: str,
        homography_matrix: np.ndarray,
        method: str = "4_point_ground",
        image_points: Optional[list[list[float]]] = None,
        ground_points_meters: Optional[list[list[float]]] = None,
        camera_height_m: float = 6.0,
        pitch_degrees: float = 35.0,
        hfov_degrees: float = 75.0,
        world_origin_gps: Optional[tuple[float, float]] = None,
        updated_at: Optional[str] = None,
    ) -> None:
        self.camera_id = camera_id
        self.H = np.asarray(homography_matrix, dtype=np.float64)
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = np.eye(3, dtype=np.float64)

        self.method = method
        self.image_points = image_points or []
        self.ground_points_meters = ground_points_meters or []
        self.camera_height_m = float(camera_height_m)
        self.pitch_degrees = float(pitch_degrees)
        self.hfov_degrees = float(hfov_degrees)
        self.world_origin_gps = world_origin_gps  # (latitude, longitude)
        self.updated_at = updated_at or datetime.now(timezone.utc).isoformat()

    def image_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """
        Project pixel coordinates (u, v) on the camera image plane
        onto ground plane coordinates (X_m, Y_m) in meters.
        """
        pt = np.array([float(u), float(v), 1.0], dtype=np.float64)
        proj = self.H @ pt
        w = proj[2]
        if abs(w) < 1e-7:
            w = 1e-7 if w >= 0 else -1e-7
        return round(float(proj[0] / w), 3), round(float(proj[1] / w), 3)

    def ground_to_image(self, x_m: float, y_m: float) -> tuple[float, float]:
        """
        Back-project ground plane coordinates (X_m, Y_m) in meters
        onto pixel coordinates (u, v) on the camera image plane.
        """
        pt = np.array([float(x_m), float(y_m), 1.0], dtype=np.float64)
        proj = self.H_inv @ pt
        w = proj[2]
        if abs(w) < 1e-7:
            w = 1e-7 if w >= 0 else -1e-7
        return round(float(proj[0] / w), 1), round(float(proj[1] / w), 1)

    def ground_to_gps(self, x_m: float, y_m: float) -> Optional[tuple[float, float]]:
        """
        Convert local ground plane coordinates (X_m East, Y_m North)
        into geodetic GPS coordinates (lat, lon).
        """
        if self.world_origin_gps is None:
            return None

        origin_lat, origin_lon = self.world_origin_gps
        # 1 degree latitude ~ 111,320 meters
        # 1 degree longitude ~ 111,320 * cos(lat) meters
        d_lat = y_m / 111320.0
        d_lon = x_m / (111320.0 * max(0.1, math.cos(math.radians(origin_lat))))
        return round(origin_lat + d_lat, 6), round(origin_lon + d_lon, 6)

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibration object to JSON dictionary."""
        return {
            "camera_id": self.camera_id,
            "method": self.method,
            "homography_matrix": self.H.tolist(),
            "image_points": self.image_points,
            "ground_points_meters": self.ground_points_meters,
            "camera_height_m": self.camera_height_m,
            "pitch_degrees": self.pitch_degrees,
            "hfov_degrees": self.hfov_degrees,
            "world_origin_gps": list(self.world_origin_gps) if self.world_origin_gps else None,
            "updated_at": self.updated_at,
        }


class CalibrationService:
    """
    Central Service for Camera Calibration, Homography Matrices,
    and 2.5D Real-World Ground Footprint Tracking.
    """

    def __init__(self, calibrations_file: Optional[Path] = None) -> None:
        self.calibrations_file = calibrations_file or DEFAULT_CALIBRATIONS_FILE
        self._calibrations: dict[str, CameraCalibration] = {}
        self.lock = threading.Lock()
        self.load()

    def load(self) -> None:
        """Load calibrations from JSON storage."""
        with self.lock:
            self._calibrations.clear()
            if not self.calibrations_file.exists():
                logger.info("No calibrations file at %s, initializing empty service.", self.calibrations_file)
                return

            try:
                with open(self.calibrations_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("calibrations", []):
                        cid = str(item.get("camera_id", "")).strip()
                        if not cid:
                            continue

                        h_mat = item.get("homography_matrix")
                        if h_mat is not None and len(h_mat) == 3:
                            h_np = np.array(h_mat, dtype=np.float64)
                        else:
                            # Reconstruct from points if matrix not directly stored
                            img_pts = item.get("image_points", [])
                            gnd_pts = item.get("ground_points_meters", [])
                            if len(img_pts) == 4 and len(gnd_pts) == 4:
                                src = np.array(img_pts, dtype=np.float32)
                                dst = np.array(gnd_pts, dtype=np.float32)
                                h_np = cv2.getPerspectiveTransform(src, dst)
                            else:
                                h_np = np.eye(3, dtype=np.float64)

                        gps = item.get("world_origin_gps")
                        origin_gps = (float(gps[0]), float(gps[1])) if gps and len(gps) == 2 else None

                        calib = CameraCalibration(
                            camera_id=cid,
                            homography_matrix=h_np,
                            method=item.get("method", "4_point_ground"),
                            image_points=item.get("image_points"),
                            ground_points_meters=item.get("ground_points_meters"),
                            camera_height_m=float(item.get("camera_height_m", 6.0)),
                            pitch_degrees=float(item.get("pitch_degrees", 35.0)),
                            hfov_degrees=float(item.get("hfov_degrees", 75.0)),
                            world_origin_gps=origin_gps,
                            updated_at=item.get("updated_at"),
                        )
                        self._calibrations[cid] = calib

                logger.info("Loaded %d camera calibrations from %s.", len(self._calibrations), self.calibrations_file)
            except Exception as err:
                logger.warning("Failed to load calibrations from %s: %s", self.calibrations_file, err)

    def save(self) -> None:
        """Persist in-memory calibrations to JSON disk storage."""
        with self.lock:
            try:
                self.calibrations_file.parent.mkdir(parents=True, exist_ok=True)
                records = [c.to_dict() for c in self._calibrations.values()]
                with open(self.calibrations_file, "w", encoding="utf-8") as f:
                    json.dump({"calibrations": records}, f, indent=2)
                logger.debug("Saved %d calibrations to %s.", len(records), self.calibrations_file)
            except Exception as err:
                logger.error("Failed to save calibrations to %s: %s", self.calibrations_file, err)

    def register_4point_calibration(
        self,
        camera_id: str,
        image_points: list[list[float]],
        ground_points_meters: list[list[float]],
        world_origin_gps: Optional[tuple[float, float]] = None,
        camera_height_m: float = 6.0,
        pitch_degrees: float = 35.0,
        hfov_degrees: float = 75.0,
    ) -> CameraCalibration:
        """
        Compute and register a 4-point perspective ground-plane homography.
        image_points: 4 non-collinear [u, v] pixel points.
        ground_points_meters: 4 corresponding [X_m, Y_m] ground plane points.
        """
        if len(image_points) != 4 or len(ground_points_meters) != 4:
            raise ValueError("Exactly 4 point correspondences are required for planar homography.")

        src = np.array(image_points, dtype=np.float32)
        dst = np.array(ground_points_meters, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        calib = CameraCalibration(
            camera_id=camera_id,
            homography_matrix=H,
            method="4_point_ground",
            image_points=image_points,
            ground_points_meters=ground_points_meters,
            camera_height_m=camera_height_m,
            pitch_degrees=pitch_degrees,
            hfov_degrees=hfov_degrees,
            world_origin_gps=world_origin_gps,
        )

        with self.lock:
            self._calibrations[camera_id] = calib

        self.save()
        logger.info("Registered 4-point ground calibration for %s.", camera_id)
        return calib

    def estimate_pinhole_calibration(
        self,
        camera_id: str,
        camera_height_m: float = 6.0,
        pitch_degrees: float = 35.0,
        hfov_degrees: float = 75.0,
        image_width: int = 1920,
        image_height: int = 1080,
        world_origin_gps: Optional[tuple[float, float]] = None,
    ) -> CameraCalibration:
        """
        Analytically synthesize homography from camera physical mounting parameters:
        height, downward tilt (pitch), and horizontal FOV.
        """
        w, h = float(image_width), float(image_height)
        theta_rad = math.radians(pitch_degrees)
        hfov_rad = math.radians(hfov_degrees)

        # Camera intrinsic focal length
        f = (w / 2.0) / math.tan(hfov_rad / 2.0)
        cx, cy = w / 2.0, h / 2.0

        # Sample 4 pixel points on the lower half of the frame (ground area)
        sample_pixels = [
            [w * 0.25, h * 0.65],
            [w * 0.75, h * 0.65],
            [w * 0.85, h * 0.95],
            [w * 0.15, h * 0.95],
        ]

        ground_meters = []
        for u, v in sample_pixels:
            # Ray vector in camera optical frame (X right, Y down, Z forward)
            ray_c = np.array([(u - cx) / f, (v - cy) / f, 1.0], dtype=np.float64)
            ray_c /= np.linalg.norm(ray_c)

            # Pitch rotation: rotate around X axis by theta (pitch downward)
            # In camera frame: pitch tilts optical axis forward-down
            # Let world coordinates be: X lateral, Y forward, Z height above ground
            # Ray direction in world frame:
            cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
            ray_w_x = ray_c[0]
            ray_w_y = ray_c[2] * cos_t - ray_c[1] * sin_t  # Forward
            ray_w_z = -(ray_c[2] * sin_t + ray_c[1] * cos_t)  # Vertical (downward is negative)

            if ray_w_z >= -1e-4:
                # Ray points upward or parallel to horizon; project to distant point
                dist = 100.0
                gx = ray_w_x * dist
                gy = dist
            else:
                # Intersection with ground plane (Z = 0) from height camera_height_m
                t = camera_height_m / (-ray_w_z)
                gx = ray_w_x * t
                gy = ray_w_y * t

            ground_meters.append([round(gx, 2), round(gy, 2)])

        src = np.array(sample_pixels, dtype=np.float32)
        dst = np.array(ground_meters, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst)

        calib = CameraCalibration(
            camera_id=camera_id,
            homography_matrix=H,
            method="pinhole_analytical",
            image_points=sample_pixels,
            ground_points_meters=ground_meters,
            camera_height_m=camera_height_m,
            pitch_degrees=pitch_degrees,
            hfov_degrees=hfov_degrees,
            world_origin_gps=world_origin_gps,
        )

        with self.lock:
            self._calibrations[camera_id] = calib

        self.save()
        logger.info("Estimated analytical pinhole calibration for %s (h=%.1fm, pitch=%.1f°).", camera_id, camera_height_m, pitch_degrees)
        return calib

    def get_calibration(self, camera_id: str) -> Optional[CameraCalibration]:
        """Retrieve active calibration for a camera ID."""
        with self.lock:
            return self._calibrations.get(str(camera_id).strip())

    def project_box_footprint(
        self,
        camera_id: str,
        box: list[float],
        frame_resolution: Optional[tuple[int, int]] = None,
    ) -> tuple[Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        """
        Extract bottom-center footpoint from 2D bounding box and project onto ground plane.

        Returns:
            (ground_meters, footprint_gps)
            ground_meters: (X_m, Y_m) relative to camera pole
            footprint_gps: (latitude, longitude)
        """
        calib = self.get_calibration(camera_id)
        if calib is None:
            return None, None

        if len(box) < 4:
            return None, None

        b = [float(v) for v in box[:4]]
        # Check if normalized coordinates (all <= 1.05)
        if max(b) <= 1.05 and frame_resolution is not None:
            w, h = frame_resolution
            x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        else:
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

        # Footpoint anchor: bottom-center of bounding box
        u_foot = (x1 + x2) / 2.0
        v_foot = y2

        ground_m = calib.image_to_ground(u_foot, v_foot)
        gps = calib.ground_to_gps(ground_m[0], ground_m[1])
        return ground_m, gps

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered camera calibrations."""
        with self.lock:
            return [c.to_dict() for c in self._calibrations.values()]


# Singleton calibration service instance
calibration_service = CalibrationService()
