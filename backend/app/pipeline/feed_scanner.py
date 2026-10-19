from __future__ import annotations

from datetime import datetime
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import cv2
import numpy as np

from app.api.cameras import _load_cameras
from app.api.stream import open_video_source
from app.pipeline.alert_service import alert_service, SNAPSHOTS_DIR
from app.pipeline.face_service import face_service
from app.pipeline.unknown_face_service import unknown_face_service

logger = logging.getLogger("SIH26187.FeedScanner")

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------
_MAX_CAMERAS = 16              # hard cap on concurrent camera workers
_INFER_WORKERS = 2             # GPU inference thread-pool size
_CAMERA_REFRESH_INTERVAL = 60  # seconds between dynamic camera list refresh
_MIN_FRAME_INTERVAL = 0.3      # fastest polling rate when faces are present (s)
_MAX_FRAME_INTERVAL = 3.0      # slowest polling rate when camera is idle (s)
_INTERVAL_RAMP_UP = 0.75       # multiplicative factor when accelerating (< 1)
_INTERVAL_RAMP_DOWN = 1.15     # multiplicative factor when decelerating (> 1)
_RECONNECT_INIT = 2.0          # first reconnect back-off delay (s)
_RECONNECT_MAX = 60.0          # maximum reconnect back-off delay (s)
_SCENE_CHANGE_THRESH = 0.03    # pixel-diff ratio to consider the scene changed
_BLUR_VAR_MIN = 20.0           # Laplacian variance below which frame is too blurry
_DARKNESS_MEAN_MIN = 15.0      # mean brightness below which frame is too dark


# ---------------------------------------------------------------------------
# _FrameBuffer — non-blocking per-camera capture thread
# ---------------------------------------------------------------------------

class _FrameBuffer:
    """
    Dedicated reader thread that keeps the single most-recent frame available
    without ever blocking the inference loop.  A Queue of maxsize=1 ensures the
    oldest frame is always discarded in favour of the newest one.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._stop_evt = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="FrameBuf")
        self._t.start()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                # Replace stale queued frame with the freshest one
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
                self._q.put(frame)
            else:
                time.sleep(0.05)

    def latest(self) -> Optional[np.ndarray]:
        """Return the most recent frame without blocking, or None if unavailable."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_evt.set()
        self._t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# ContinuousFaceScanner
# ---------------------------------------------------------------------------

class ContinuousFaceScanner:
    """
    Continuous Multi-Feed Background Video Scanner for MAATRIX.

    Monitors registered camera streams, periodically decodes frames,
    performs UniFace (512-D) biometric detection against enrolled suspects,
    and automatically triggers real-time security alerts with snapshot forensics.

    Improvements over v1
    --------------------
    1. Non-blocking frame capture   — _FrameBuffer thread per camera
    2. Scene-change gating          — skip inference on static frames
    3. Blur / darkness rejection    — skip unusable frames early
    4. Shared inference pool        — decouple I/O from GPU work
    5. Adaptive frame interval      — faster when active, slower when idle
    6. Exponential reconnect        — back-off up to 60 s on dead streams
    7. Async alert dispatch         — alert I/O never blocks inference loop
    8. Dynamic camera refresh       — pick up new / removed cameras at runtime
    """

    def __init__(self, frame_interval: float = 0.6,
                 cooldown_seconds: float = 60.0) -> None:
        self.frame_interval = frame_interval
        self.cooldown_seconds = cooldown_seconds
        self.running = False
        self.lock = threading.Lock()

        # Thread registry: cam_id -> Thread
        self.threads: dict[str, threading.Thread] = {}
        self.worker_stop_events: dict[str, threading.Event] = {}

        # Frame buffers: cam_id -> _FrameBuffer (kept for clean shutdown)
        self._frame_buffers: dict[str, _FrameBuffer] = {}

        # Debounce dictionary: (suspect_name, camera_id) -> timestamp
        self.last_alert_time: dict[tuple[str, str], float] = {}

        # Telemetry metrics
        self.total_scans_performed = 0
        self.total_suspects_found = 0
        self.last_scan_time: Optional[str] = None

        # Adaptive per-camera polling intervals: cam_id -> current interval (s)
        self._cam_intervals: dict[str, float] = {}

        # Shared inference thread-pool (improvement #4)
        self._infer_pool = ThreadPoolExecutor(
            max_workers=_INFER_WORKERS,
            thread_name_prefix="ScanInfer",
        )

        # Dynamic camera refresh thread (improvement #8)
        self._refresh_stop_evt: Optional[threading.Event] = None
        self._refresh_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> dict[str, Any]:
        """Start continuous scanning across all active camera feeds."""
        with self.lock:
            if self.running:
                return {"success": True, "message": "Scanner is already running"}
            self.running = True

        cameras = _load_cameras()
        active_cameras = [
            c for c in cameras
            if c.get("alertTriggerEnabled", True)
        ][:_MAX_CAMERAS]

        logger.info(
            "Starting ContinuousFaceScanner across %d active camera feeds...",
            len(active_cameras),
        )

        self.worker_stop_events = {}
        self.threads = {}
        self._frame_buffers = {}

        for cam in active_cameras:
            self._spawn_worker(cam)

        # Start dynamic camera refresh watcher
        self._refresh_stop_evt = threading.Event()
        self._refresh_thread = threading.Thread(
            target=self._camera_refresh_worker,
            args=(self._refresh_stop_evt,),
            name="CamRefresh",
            daemon=True,
        )
        self._refresh_thread.start()

        return {
            "success": True,
            "message": (
                f"Continuous Face Scanner started with {len(self.threads)} feed workers."
            ),
            "monitored_cameras": [c.get("name") for c in active_cameras],
        }

    def stop(self) -> dict[str, Any]:
        """Gracefully stop all camera feed scanner worker threads."""
        with self.lock:
            if not self.running:
                return {"success": True, "message": "Scanner is not running"}
            self.running = False

        logger.info("Stopping ContinuousFaceScanner worker threads...")

        # Stop dynamic refresh watcher
        if self._refresh_stop_evt:
            self._refresh_stop_evt.set()
        if self._refresh_thread:
            self._refresh_thread.join(timeout=2.0)

        for cam_id in list(self.worker_stop_events.keys()):
            self._stop_worker(cam_id)

        self._infer_pool.shutdown(wait=False)
        logger.info("All ContinuousFaceScanner worker threads stopped.")
        return {"success": True, "message": "All camera feed scanner workers stopped."}

    def get_status(self) -> dict[str, Any]:
        """Return operational telemetry for the feed scanner."""
        with self.lock:
            active_tids = [cid for cid, t in self.threads.items() if t.is_alive()]
            intervals = dict(self._cam_intervals)
        return {
            "running": self.running,
            "worker_count": len(active_tids),
            "active_camera_ids": active_tids,
            "total_scans": self.total_scans_performed,
            "suspect_detections": self.total_suspects_found,
            "last_scan_time": self.last_scan_time or "None",
            "sampling_interval_sec": self.frame_interval,
            "debounce_cooldown_sec": self.cooldown_seconds,
            "adaptive_intervals": intervals,
        }

    # ------------------------------------------------------------------
    # Worker lifecycle helpers
    # ------------------------------------------------------------------

    def _spawn_worker(self, camera: dict[str, Any]) -> None:
        """Create and start a scanner worker thread for a single camera."""
        cam_id = str(camera.get("id"))
        if cam_id in self.threads and self.threads[cam_id].is_alive():
            return  # already running

        stop_evt = threading.Event()
        self.worker_stop_events[cam_id] = stop_evt
        self._cam_intervals.setdefault(cam_id, self.frame_interval)

        t = threading.Thread(
            target=self._scan_camera_worker,
            args=(camera, stop_evt),
            name=f"FeedScanner-{cam_id}",
            daemon=True,
        )
        self.threads[cam_id] = t
        t.start()
        logger.info("Spawned scanner worker for camera %s", cam_id)

    def _stop_worker(self, cam_id: str) -> None:
        """Signal and join a single camera worker thread."""
        evt = self.worker_stop_events.pop(cam_id, None)
        if evt:
            evt.set()
        t = self.threads.pop(cam_id, None)
        if t:
            t.join(timeout=1.0)
        buf = self._frame_buffers.pop(cam_id, None)
        if buf:
            buf.stop()
        self._cam_intervals.pop(cam_id, None)
        logger.info("Stopped scanner worker for camera %s", cam_id)

    # ------------------------------------------------------------------
    # Improvement #8 — Dynamic camera refresh
    # ------------------------------------------------------------------

    def _camera_refresh_worker(self, stop_evt: threading.Event) -> None:
        """
        Periodically re-reads the camera list and spawns/stops workers
        so newly added cameras are monitored without a full scanner restart.
        """
        while not stop_evt.wait(_CAMERA_REFRESH_INTERVAL):
            if not self.running:
                break
            try:
                cameras = _load_cameras()
                desired: dict[str, dict[str, Any]] = {
                    str(c["id"]): c
                    for c in cameras
                    if c.get("alertTriggerEnabled", True)
                }

                with self.lock:
                    current_ids = set(self.threads.keys())

                new_ids = set(desired.keys()) - current_ids
                removed_ids = current_ids - set(desired.keys())

                for cam_id in new_ids:
                    if len(self.threads) < _MAX_CAMERAS:
                        self._spawn_worker(desired[cam_id])
                        logger.info(
                            "CamRefresh: added new camera worker %s", cam_id
                        )

                for cam_id in removed_ids:
                    self._stop_worker(cam_id)
                    logger.info(
                        "CamRefresh: removed stale camera worker %s", cam_id
                    )

            except Exception as refresh_err:
                logger.debug("CamRefresh error: %s", refresh_err)

    # ------------------------------------------------------------------
    # Improvement #3 — Frame-level scene-change detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_scene_changed(
        prev: Optional[np.ndarray],
        curr: np.ndarray,
        threshold: float = _SCENE_CHANGE_THRESH,
    ) -> bool:
        """
        Return True if enough pixels changed between frames to warrant inference.
        Uses a cheap 160×90 downscale + absdiff — sub-millisecond cost.
        """
        if prev is None:
            return True
        try:
            prev_g = cv2.cvtColor(cv2.resize(prev, (160, 90)), cv2.COLOR_BGR2GRAY)
            curr_g = cv2.cvtColor(cv2.resize(curr, (160, 90)), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_g, curr_g)
            ratio = float(np.count_nonzero(diff > 25)) / diff.size
            return ratio > threshold
        except Exception:
            return True  # fail-open: always infer if comparison fails

    # ------------------------------------------------------------------
    # Improvement #4 — Blur & darkness rejection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_frame_usable(frame: np.ndarray) -> bool:
        """
        Return False if the frame is too blurry (Laplacian variance < threshold)
        or too dark (mean brightness < threshold).  Saves GPU cycles and prevents
        ghost detections from motion-blurred or unlit frames.
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < _BLUR_VAR_MIN:
                return False
            if float(gray.mean()) < _DARKNESS_MEAN_MIN:
                return False
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # Improvement #5 — Adaptive frame interval
    # ------------------------------------------------------------------

    def _adjust_interval(self, cam_id: str, faces_found: int) -> None:
        """
        Accelerate polling when faces are detected; decelerate when the scene is
        idle.  Converges toward _MIN or _MAX asymptotically.
        """
        current = self._cam_intervals.get(cam_id, self.frame_interval)
        if faces_found > 0:
            new_interval = max(_MIN_FRAME_INTERVAL, current * _INTERVAL_RAMP_UP)
        else:
            new_interval = min(_MAX_FRAME_INTERVAL, current * _INTERVAL_RAMP_DOWN)
        self._cam_intervals[cam_id] = new_interval

    # ------------------------------------------------------------------
    # Improvement #7 — Async alert dispatch
    # ------------------------------------------------------------------

    def _dispatch_alert_async(self, **kwargs: Any) -> None:
        """Fire alert creation on a daemon thread so the inference loop never
        blocks on disk / DB I/O."""
        threading.Thread(
            target=self._do_dispatch_alert,
            kwargs=kwargs,
            daemon=True,
            name="AlertDispatch",
        ).start()

    @staticmethod
    def _do_dispatch_alert(**kwargs: Any) -> None:
        try:
            alert_service.create_alert(**kwargs)
        except Exception as dispatch_err:
            logger.warning("Alert dispatch error: %s", dispatch_err)

    # ------------------------------------------------------------------
    # Core camera worker
    # ------------------------------------------------------------------

    def _scan_camera_worker(
        self, camera: dict[str, Any], stop_evt: threading.Event
    ) -> None:
        """
        Dedicated worker loop for an individual camera stream.

        Lifecycle
        ---------
        1. Open stream with exponential reconnect back-off (improvement #6).
        2. Attach a _FrameBuffer for non-blocking I/O (improvement #1).
        3. On each tick: scene-change gate → blur/darkness gate → GPU inference
           (improvements #2, #3, #4 via shared pool).
        4. Adjust interval adaptively (improvement #5).
        5. Dispatch alerts asynchronously (improvement #7).
        """
        cam_id = str(camera.get("id", "cam-unknown"))
        cam_name = str(camera.get("name", f"Camera {cam_id}"))
        cam_location = str(camera.get("location", "Perimeter"))
        stream_url = str(camera.get("streamUrl", "sample"))

        logger.info(
            "Worker thread initialized for camera '%s' (%s)", cam_name, cam_id
        )

        cap: Optional[cv2.VideoCapture] = None
        frame_buf: Optional[_FrameBuffer] = None
        last_frame: Optional[np.ndarray] = None
        reconnect_backoff = _RECONNECT_INIT

        while self.running and not stop_evt.is_set():
            t0 = time.perf_counter()

            # ----------------------------------------------------------------
            # Improvement #6 — Exponential reconnect back-off
            # ----------------------------------------------------------------
            if cap is None or not cap.isOpened():
                # Tear down stale frame buffer if present
                if frame_buf is not None:
                    frame_buf.stop()
                    frame_buf = None
                if cam_id in self._frame_buffers:
                    del self._frame_buffers[cam_id]

                try:
                    cap, _diag, _, _ = open_video_source(stream_url)
                    if not cap or not cap.isOpened():
                        raise RuntimeError("VideoCapture not opened")
                    reconnect_backoff = _RECONNECT_INIT  # reset on success

                    # Improvement #1 — attach non-blocking frame buffer
                    frame_buf = _FrameBuffer(cap)
                    self._frame_buffers[cam_id] = frame_buf

                except Exception as open_err:
                    logger.debug(
                        "Scanner could not open %s (%s): %s — retrying in %.0fs",
                        cam_name, stream_url, open_err, reconnect_backoff,
                    )
                    stop_evt.wait(reconnect_backoff)
                    reconnect_backoff = min(reconnect_backoff * 2, _RECONNECT_MAX)
                    cap = None
                    continue

            # ----------------------------------------------------------------
            # Improvement #1 — grab latest frame from non-blocking buffer
            # ----------------------------------------------------------------
            frame = frame_buf.latest() if frame_buf else None
            if frame is None:
                # No new frame available yet; wait a short interval
                stop_evt.wait(0.05)
                continue

            # ----------------------------------------------------------------
            # Improvement #3 — skip static scenes
            # ----------------------------------------------------------------
            if not self._is_scene_changed(last_frame, frame):
                self._adjust_interval(cam_id, 0)
                stop_evt.wait(self._cam_intervals.get(cam_id, self.frame_interval))
                continue

            # ----------------------------------------------------------------
            # Improvement #4 — skip blurry / dark frames
            # ----------------------------------------------------------------
            if not self._is_frame_usable(frame):
                logger.debug(
                    "Camera %s: frame rejected (blur/dark) — skipping inference",
                    cam_name,
                )
                stop_evt.wait(self._cam_intervals.get(cam_id, self.frame_interval))
                continue

            last_frame = frame

            # Update telemetry
            with self.lock:
                self.total_scans_performed += 1
                self.last_scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Preserve high-resolution details for distant CCTV faces (up to 1280 px)
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280.0 / w
                infer_frame = cv2.resize(frame, (1280, int(h * scale)))
            else:
                infer_frame = frame

            # ----------------------------------------------------------------
            # Improvement #2 — submit inference to shared thread-pool
            # ----------------------------------------------------------------
            future = self._infer_pool.submit(
                self._run_inference,
                cam_id, cam_name, cam_location, infer_frame,
            )
            future.add_done_callback(self._handle_infer_result)

            # ----------------------------------------------------------------
            # Improvement #5 — adaptive interval (will be refined in callback)
            # ----------------------------------------------------------------
            elapsed = time.perf_counter() - t0
            interval = self._cam_intervals.get(cam_id, self.frame_interval)
            sleep_time = max(0.05, interval - elapsed)
            stop_evt.wait(sleep_time)

        # Clean up on exit
        if frame_buf is not None:
            frame_buf.stop()
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        logger.info("Worker thread exited for camera '%s' (%s)", cam_name, cam_id)

    # ------------------------------------------------------------------
    # Inference task (runs inside _infer_pool threads)
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        cam_id: str,
        cam_name: str,
        cam_location: str,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        """
        Execute face detection + recognition on the pool thread and return a
        result bundle.  Any exception is captured so the callback can log it.
        """
        try:
            # Run UniFace recognition with surveillance-calibrated parameters
            faces = face_service.detect_and_recognize(
                frame,
                min_match_score=None,
                min_face_size=12,
                det_score_thresh=0.25,
                use_temporal_smoothing=False,
                enable_distant_mode=True,
            )
            return {
                "ok": True,
                "cam_id": cam_id,
                "cam_name": cam_name,
                "cam_location": cam_location,
                "frame": frame,
                "faces": faces,
            }
        except Exception as infer_err:
            return {"ok": False, "cam_id": cam_id, "cam_name": cam_name,
                    "error": str(infer_err)}

    def _handle_infer_result(self, future: Any) -> None:
        """
        Callback executed when an inference future completes.
        Checks for suspects and dispatches alerts asynchronously.
        """
        try:
            result = future.result()
        except Exception as fut_err:
            logger.debug("Inference future raised: %s", fut_err)
            return

        if not result.get("ok"):
            logger.debug(
                "Inference error on %s: %s",
                result.get("cam_name"), result.get("error"),
            )
            return

        cam_id = result["cam_id"]
        cam_name = result["cam_name"]
        cam_location = result["cam_location"]
        infer_frame = result["frame"]
        faces: list[dict[str, Any]] = result["faces"]

        # Improvement #5 — adapt polling rate based on detection count
        self._adjust_interval(cam_id, len(faces))

        now_sec = time.time()

        for face in faces:
            is_known = bool(face.get("is_known", False))
            if not is_known:
                # Capture and record unknown person with real timestamp and photograph
                try:
                    unknown_face_service.record_unknown(
                        frame=infer_frame,
                        face=face,
                        camera_id=cam_id,
                        camera_name=cam_name,
                    )
                except Exception as unk_err:
                    logger.debug("Unknown record error in feed scanner: %s", unk_err)
                continue

            if not face.get("is_threat"):
                continue

            suspect_name = face["name"]
            threat_lvl = face.get("threat_level", "HIGH")
            category = face.get("category", "Wanted / BOLO")
            calibrated_conf = face.get("calibrated_conf", 85.0)

            # Debounce check
            debounce_key = (suspect_name.lower(), cam_id)
            last_time = self.last_alert_time.get(debounce_key, 0.0)
            if now_sec - last_time < self.cooldown_seconds:
                continue

            self.last_alert_time[debounce_key] = now_sec
            with self.lock:
                self.total_suspects_found += 1

            logger.warning(
                "🚨 SUSPECT MATCH DETECTED on %s (%s): %s [%s] (Conf: %.1f%%)",
                cam_name, cam_id, suspect_name, threat_lvl, calibrated_conf,
            )

            # Annotate frame snapshot
            annotated_snapshot = face_service.draw_faces(infer_frame, [face])

            # Save snapshot JPEG
            ts_str = int(now_sec * 1000)
            snapshot_name = f"alert_{ts_str}_{cam_id}.jpg"
            snapshot_path = SNAPSHOTS_DIR / snapshot_name
            try:
                cv2.imwrite(str(snapshot_path), annotated_snapshot)
            except Exception as save_err:
                logger.warning("Could not save snapshot file: %s", save_err)
                snapshot_name = None

            # Improvement #7 — async alert dispatch
            title = f"🚨 SUSPECT SIGHTING: {suspect_name.upper()} [{threat_lvl}]"
            self._dispatch_alert_async(
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


feed_scanner_service = ContinuousFaceScanner()
