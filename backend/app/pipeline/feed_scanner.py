from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from app.api.cameras import _load_cameras
from app.api.stream import open_video_source
from app.pipeline.alert_service import alert_service, SNAPSHOTS_DIR
from app.pipeline.face_service import face_service

logger = logging.getLogger("SIH26187.FeedScanner")


class ContinuousFaceScanner:
    """
    Continuous Multi-Feed Background Video Scanner for MAATRIX.
    Monitors registered camera streams, periodically decodes frames,
    performs YuNet + SFace biometric detection against enrolled suspects,
    and automatically triggers real-time security alerts with snapshot forensics.
    """

    def __init__(self, frame_interval: float = 0.6, cooldown_seconds: float = 60.0) -> None:
        self.frame_interval = frame_interval
        self.cooldown_seconds = cooldown_seconds
        self.running = False
        self.lock = threading.Lock()

        # Thread registry: cam_id -> Thread
        self.threads: dict[str, threading.Thread] = {}
        self.worker_stop_events: dict[str, threading.Event] = {}

        # Debounce dictionary: (suspect_name, camera_id) -> timestamp
        self.last_alert_time: dict[tuple[str, str], float] = {}

        # Telemetry metrics
        self.total_scans_performed = 0
        self.total_suspects_found = 0
        self.last_scan_time: Optional[str] = None

    def start(self) -> dict[str, Any]:
        """Start continuous scanning across all active camera feeds."""
        with self.lock:
            if self.running:
                return {"success": True, "message": "Scanner is already running"}
            self.running = True

        cameras = _load_cameras()
        # Find cameras with alertTriggerEnabled == True (limit to max 8 concurrent threads to prevent resource starvation)
        active_cameras = [
            c for c in cameras
            if c.get("alertTriggerEnabled", True)
        ][:8]

        logger.info("Starting ContinuousFaceScanner across %d active camera feeds...", len(active_cameras))

        self.worker_stop_events = {}
        self.threads = {}

        for cam in active_cameras:
            cam_id = str(cam.get("id"))
            stop_evt = threading.Event()
            self.worker_stop_events[cam_id] = stop_evt

            t = threading.Thread(
                target=self._scan_camera_worker,
                args=(cam, stop_evt),
                name=f"FeedScanner-{cam_id}",
                daemon=True,
            )
            self.threads[cam_id] = t
            t.start()

        return {
            "success": True,
            "message": f"Continuous Face Scanner started with {len(self.threads)} feed workers.",
            "monitored_cameras": [c.get("name") for c in active_cameras],
        }

    def stop(self) -> dict[str, Any]:
        """Gracefully stop all camera feed scanner worker threads."""
        with self.lock:
            if not self.running:
                return {"success": True, "message": "Scanner is not running"}
            self.running = False

        logger.info("Stopping ContinuousFaceScanner worker threads...")
        for cam_id, stop_evt in list(self.worker_stop_events.items()):
            stop_evt.set()

        for cam_id, t in list(self.threads.items()):
            t.join(timeout=1.0)

        self.threads.clear()
        self.worker_stop_events.clear()
        logger.info("All ContinuousFaceScanner worker threads stopped.")

        return {"success": True, "message": "All camera feed scanner workers stopped."}

    def get_status(self) -> dict[str, Any]:
        """Return operational telemetry for the feed scanner."""
        with self.lock:
            active_tids = [cid for cid, t in self.threads.items() if t.is_alive()]
            return {
                "running": self.running,
                "worker_count": len(active_tids),
                "active_camera_ids": active_tids,
                "total_scans": self.total_scans_performed,
                "suspect_detections": self.total_suspects_found,
                "last_scan_time": self.last_scan_time or "None",
                "sampling_interval_sec": self.frame_interval,
                "debounce_cooldown_sec": self.cooldown_seconds,
            }

    def _scan_camera_worker(self, camera: dict[str, Any], stop_evt: threading.Event) -> None:
        """Dedicated worker loop for an individual camera stream."""
        cam_id = str(camera.get("id", "cam-unknown"))
        cam_name = str(camera.get("name", f"Camera {cam_id}"))
        cam_location = str(camera.get("location", "Perimeter"))
        stream_url = str(camera.get("streamUrl", "sample"))

        logger.info("Worker thread initialized for camera '%s' (%s)", cam_name, cam_id)
        cap: Optional[cv2.VideoCapture] = None
        consecutive_errors = 0

        while self.running and not stop_evt.is_set():
            t0 = time.perf_counter()

            # Ensure camera capture is open
            if cap is None or not cap.isOpened():
                try:
                    cap, diag, _, _ = open_video_source(stream_url)
                    if not cap or not cap.isOpened():
                        # If feed unreachable, sleep 5 seconds before retrying
                        stop_evt.wait(5.0)
                        continue
                except Exception as open_err:
                    logger.debug("Scanner could not open %s (%s): %s", cam_name, stream_url, open_err)
                    stop_evt.wait(5.0)
                    continue

            try:
                success, frame = cap.read()
                if not success or frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors > 5:
                        if cap:
                            cap.release()
                            cap = None
                        consecutive_errors = 0
                        stop_evt.wait(2.0)
                        continue
                    stop_evt.wait(0.2)
                    continue

                consecutive_errors = 0

                # Decimated frame processing
                with self.lock:
                    self.total_scans_performed += 1
                    self.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Preserve high-resolution details for distant CCTV faces (up to 1280px)
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280.0 / w
                    small_frame = cv2.resize(frame, (1280, int(h * scale)))
                else:
                    small_frame = frame

                # Run YuNet + SFace recognition with surveillance-calibrated parameters
                faces = face_service.detect_and_recognize(
                    small_frame,
                    min_match_score=0.34,
                    min_face_size=24,
                    det_score_thresh=0.45,
                    use_temporal_smoothing=False,
                )

                # Inspect detected faces for suspects
                for face in faces:
                    if face.get("is_known") and face.get("is_threat"):
                        suspect_name = face["name"]
                        threat_lvl = face.get("threat_level", "HIGH")
                        category = face.get("category", "Wanted / BOLO")
                        calibrated_conf = face.get("calibrated_conf", 85.0)

                        # Check debounce: (suspect_name, cam_id)
                        debounce_key = (suspect_name.lower(), cam_id)
                        now_sec = time.time()
                        last_time = self.last_alert_time.get(debounce_key, 0.0)

                        if now_sec - last_time >= self.cooldown_seconds:
                            self.last_alert_time[debounce_key] = now_sec
                            with self.lock:
                                self.total_suspects_found += 1

                            logger.warning(
                                "🚨 SUSPECT MATCH DETECTED on %s (%s): %s [%s] (Conf: %.1f%%)",
                                cam_name,
                                cam_id,
                                suspect_name,
                                threat_lvl,
                                calibrated_conf,
                            )

                            # Annotate frame with high-visibility red box & details
                            annotated_snapshot = face_service.draw_faces(small_frame, [face])

                            # Save alert snapshot JPEG
                            ts_str = int(now_sec * 1000)
                            snapshot_name = f"alert_{ts_str}_{cam_id}.jpg"
                            snapshot_path = SNAPSHOTS_DIR / snapshot_name
                            try:
                                cv2.imwrite(str(snapshot_path), annotated_snapshot)
                            except Exception as save_err:
                                logger.warning("Could not save snapshot file: %s", save_err)
                                snapshot_name = None

                            # Dispatch alert
                            title = f"🚨 SUSPECT SIGHTING: {suspect_name.upper()} [{threat_lvl}]"
                            alert_service.create_alert(
                                title=title,
                                location=f"{cam_name} — {cam_location}",
                                severity="High" if threat_lvl in ("CRITICAL", "HIGH") else "Medium",
                                camera_id=cam_id,
                                camera_name=cam_name,
                                class_name="suspect",
                                confidence=calibrated_conf / 100.0,
                                box=face.get("bbox", []),
                                snapshot_filename=snapshot_name,
                                suspect_name=suspect_name,
                                threat_level=threat_lvl,
                                category=category,
                                notes=f"Automatic biometric match on feed {cam_name}",
                            )

            except Exception as loop_err:
                logger.debug("Scanner loop exception on camera %s: %s", cam_name, loop_err)
                stop_evt.wait(0.5)

            # Enforce sampling interval (e.g. ~1.2 seconds between checks on this camera)
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.1, self.frame_interval - elapsed)
            stop_evt.wait(sleep_time)

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        logger.info("Worker thread exited for camera '%s' (%s)", cam_name, cam_id)


feed_scanner_service = ContinuousFaceScanner()
