from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import os
from pathlib import Path
import threading
import time
from typing import Generator, Optional
import uuid

import cv2
from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse
import numpy as np

from app.core.config import settings
from app.core.runtime import inference_service, model_manager
from app.pipeline.preprocessor import PreprocessingConfig, Preprocessor

logger = logging.getLogger("SIH26187.Stream")

router = APIRouter(
    prefix="/api/stream",
    tags=["Stream"],
)

# ── Session-based stream stop & real-time telemetry registries ─────────────────
# Maps session_id → threading.Event.  Each live stream connection registers
# here.  When the frontend navigates away it calls
# DELETE /api/stream/live/stop?session_id=X, which sets the event and causes
# the generator to exit its while-True loop within one frame interval (~40 ms),
# freeing all GPU/CPU inference resources immediately.
_live_stop_events: dict[str, threading.Event] = {}
# Maps session_id → latest structured detection telemetry (targets, faces, vehicles, fps, latency)
_live_session_detections: dict[str, dict] = {}

# Colors for bounding box classes (BGR)
CLASS_COLORS = {
    "person": (0, 220, 100),         # Emerald
    "firearm": (50, 50, 255),        # Red
    "explosive": (0, 0, 255),        # Deep Red
    "melee_weapon": (30, 100, 255),  # Orange-Red
    "blunt_weapon": (0, 165, 255),   # Orange
    "tool": (255, 190, 0),           # Cyan-Blue
    "fire_smoke": (0, 140, 255),     # Deep Amber
    "car": (255, 200, 0),            # Yellow
    "vehicle": (255, 200, 0),        # Yellow
    "motorcycle": (255, 160, 0),     # Amber
    "bus": (255, 180, 20),           # Amber-Orange
    "truck": (255, 140, 0),          # Orange
}
DEFAULT_COLOR = (0, 255, 180)

SAMPLE_VIDEO_PATH = settings.model_directory.parent / "test_video_input.mp4"


def create_standby_frame(
    rtsp_url: str,
    status_msg: str = "Connecting to Live Stream...",
    protocol: str = "LIVE STREAM",
    width: int = 854,
    height: int = 480,
) -> np.ndarray:
    """Generate a tactical dark placeholder frame with status telemetry."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 22, 26)  # Dark slate background

    # Grid scanlines
    for y in range(0, height, 20):
        cv2.line(frame, (0, y), (width, y), (26, 32, 38), 1)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (26, 32, 38), 1)

    # Tactical corner markers
    cv2.line(frame, (20, 20), (50, 20), (0, 200, 120), 2)
    cv2.line(frame, (20, 20), (20, 50), (0, 200, 120), 2)
    cv2.line(frame, (width - 20, 20), (width - 50, 20), (0, 200, 120), 2)
    cv2.line(frame, (width - 20, 20), (width - 20, 50), (0, 200, 120), 2)
    cv2.line(frame, (20, height - 20), (50, height - 20), (0, 200, 120), 2)
    cv2.line(frame, (20, height - 20), (20, height - 50), (0, 200, 120), 2)
    cv2.line(frame, (width - 20, height - 20), (width - 50, height - 20), (0, 200, 120), 2)
    cv2.line(frame, (width - 20, height - 20), (width - 20, height - 50), (0, 200, 120), 2)

    # Header
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_title = f"MAATRIX • {protocol.upper()}"
    cv2.putText(frame, header_title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 120), 2)
    cv2.putText(frame, f"TIMESTAMP: {now_str} IST", (width - 280, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 175, 190), 1)

    # Center Alert Box
    center_y = height // 2
    cv2.rectangle(frame, (width // 2 - 300, center_y - 60), (width // 2 + 300, center_y + 60), (35, 45, 55), -1)
    cv2.rectangle(frame, (width // 2 - 300, center_y - 60), (width // 2 + 300, center_y + 60), (50, 70, 90), 1)

    # Blinking status dot
    blink = int(time.time() * 2) % 2 == 0
    dot_color = (0, 200, 120) if blink else (0, 100, 60)
    cv2.circle(frame, (width // 2 - 260, center_y - 15), 8, dot_color, -1)

    cv2.putText(frame, status_msg[:50], (width // 2 - 230, center_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 250), 2)
    display_url = rtsp_url if len(rtsp_url) <= 55 else rtsp_url[:52] + "..."
    cv2.putText(frame, f"Feed Source: {display_url}", (width // 2 - 260, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 160, 180), 1)

    # Bottom status
    cv2.putText(frame, f"STATUS: {status_msg[:75]}", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 160), 1)
    return frame


import socket
from urllib.parse import urlparse


def is_webcam_source(raw_url: str) -> bool:
    """Check if stream URL points to a local hardware webcam device index or alias."""
    clean = (raw_url or "").strip().lower()
    if clean.isdigit():
        return True
    if clean in ("webcam", "camera", "default", "usb", "cam", "device0", "device 0"):
        return True
    if clean.startswith("webcam:") or clean.startswith("camera:"):
        return True
    return False


def get_webcam_index(raw_url: str) -> int:
    """Extract integer camera device index from webcam string (e.g. '0', 'webcam:1' -> 1)."""
    clean = (raw_url or "").strip().lower()
    if clean.isdigit():
        return int(clean)
    if ":" in clean:
        parts = clean.split(":")
        if len(parts) > 1 and parts[1].strip().isdigit():
            return int(parts[1].strip())
    return 0


def normalize_stream_url(raw_url: str) -> tuple[str, list[str], str]:
    """
    Normalizes any video source input (Webcam, RTSP, HTTP IP Webcam, bare host:port, or sample).
    Returns:
        canonical_url: Primary sanitized URL
        candidates: List of fallback candidate URLs to test (crucial for IP Webcam Pro / DroidCam)
        protocol_name: Human-friendly protocol descriptor
    """
    clean = (raw_url or "").strip()
    if not clean or clean.lower() in ("sample", "demo", "test"):
        return "sample", ["sample"], "Sample Video Simulation"

    if is_webcam_source(clean):
        idx = get_webcam_index(clean)
        return str(idx), [str(idx)], f"Local Webcam (Device {idx})"

    if Path(clean).is_file():
        return clean, [clean], "Local Video Source"

    # Detect scheme if omitted (e.g. 12.10.5.194:8080 or 192.168.1.100:554/live)
    if "://" not in clean:
        if ":554" in clean or "rtsp" in clean.lower():
            clean = f"rtsp://{clean}"
        else:
            clean = f"http://{clean}"

    scheme = clean.split("://")[0].lower()

    if scheme in ("rtsp", "rtsps"):
        return clean, [clean], "RTSP Live Stream"

    if scheme in ("http", "https"):
        parsed = urlparse(clean)
        path = parsed.path.rstrip("/")
        port = parsed.port or (443 if scheme == "https" else 80)
        base_origin = f"{parsed.scheme}://{parsed.netloc}"

        candidates = []
        if path and path not in ("", "/"):
            # User gave specific path (e.g. /video or /videofeed or /mjpeg)
            candidates.append(clean)
            if not path.endswith("/video"):
                candidates.append(f"{base_origin}/video")
            if not path.endswith("/videofeed"):
                candidates.append(f"{base_origin}/videofeed")
        else:
            # User provided root URL e.g. http://12.10.5.194:8080 or http://192.168.1.10:8080/
            # For IP Webcam / IP Webcam Pro, video feed is served at /video or /videofeed
            candidates.append(f"{base_origin}/video")
            candidates.append(f"{base_origin}/videofeed")
            candidates.append(clean)

        protocol_label = "IP Webcam Pro (HTTP)" if port in (8080, 8081, 4747) else "HTTP Camera Stream"
        return candidates[0], candidates, protocol_label

    return clean, [clean], f"Network Stream ({scheme})"


def is_host_reachable(url: str, timeout_seconds: float = 1.5) -> tuple[bool, str, str, int]:
    """Fast probe to verify host reachability or local camera device accessibility."""
    clean = url.strip()
    if is_webcam_source(clean):
        idx = get_webcam_index(clean)
        return True, f"Local Webcam Device {idx} ready", "localhost", idx

    if "://" not in clean:
        clean = f"http://{clean}"
    parsed = urlparse(clean)
    host = parsed.hostname or "127.0.0.1"
    scheme = (parsed.scheme or "http").lower()
    default_port = {"rtsp": 554, "rtsps": 322, "http": 80, "https": 443}.get(scheme, 80)
    port = parsed.port or default_port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout_seconds)
        t0 = time.perf_counter()
        sock.connect((host, port))
        latency = int((time.perf_counter() - t0) * 1000)
        sock.close()
        return True, f"Host reachable at {host}:{port} ({latency}ms)", host, port
    except socket.timeout:
        return False, f"Connection timed out reaching {host}:{port} (Device unreachable on current network/Wi-Fi)", host, port
    except ConnectionRefusedError:
        return False, f"Connection refused at {host}:{port} (Server not running on this port)", host, port
    except Exception as exc:
        return False, f"Unreachable at {host}:{port} ({exc})", host, port


def is_rtsp_host_reachable(rtsp_url: str, timeout_seconds: float = 1.0) -> tuple[bool, str]:
    """Backward-compatible probe wrapper."""
    reachable, diag, _, _ = is_host_reachable(rtsp_url, timeout_seconds)
    return reachable, diag


def open_video_source(source_url: str) -> tuple[cv2.VideoCapture | None, str, str, str]:
    """
    Open any supported video source: Local Webcam (0/1), RTSP, HTTP IP Webcam Pro, local file, or sample demo.
    Returns:
        (cap, diag_message, resolved_url, protocol_name)
    """
    clean = source_url.strip()
    canonical_url, candidates, protocol = normalize_stream_url(clean)

    # --- Sample / demo keyword ---
    if canonical_url == "sample":
        if SAMPLE_VIDEO_PATH.exists():
            cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
            if cap.isOpened():
                return cap, "Sample Surveillance Feed Active", str(SAMPLE_VIDEO_PATH), protocol
        return None, "Sample video file not found on server", "sample", protocol

    # --- Local Webcam Device ---
    if is_webcam_source(canonical_url):
        idx = get_webcam_index(canonical_url)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if cap.isOpened():
            success, frame = cap.read()
            if success and frame is not None and frame.size > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap, f"Webcam Device {idx} Connected", str(idx), f"Local Webcam (Device {idx})"
            cap.release()
        return None, f"Could not access Webcam Device {idx}", str(idx), f"Local Webcam (Device {idx})"

    # --- Local file ---
    if Path(canonical_url).is_file():
        cap = cv2.VideoCapture(canonical_url)
        if cap.isOpened():
            return cap, "Local Video File Active", canonical_url, protocol
        return None, f"Could not read local file: {canonical_url}", canonical_url, protocol

    # --- Network Probe ---
    reachable, diag, host, port = is_host_reachable(canonical_url, timeout_seconds=1.5)
    if not reachable:
        logger.info("Host pre-check failed for %s: %s", canonical_url, diag)
        return None, diag, canonical_url, protocol

    # --- Network Stream Opening with Candidate Trial ---
    scheme = canonical_url.split("://")[0].lower() if "://" in canonical_url else ""

    # Configure low-latency FFmpeg capture options:
    # - rtsp_transport;tcp: prevents UDP packet drop artifacts
    # - fflags;nobuffer: disables FFmpeg internal packet buffering to stop latency accumulation
    # - flags;low_delay: forces zero-delay decoding in libavcodec
    # - max_delay;500000: caps maximum network jitter buffer to 500ms
    if scheme in ("rtsp", "rtsps"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
        )
    elif scheme in ("http", "https"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "fflags;nobuffer|flags;low_delay|max_delay;500000"
        )

    for candidate in candidates:
        try:
            cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
            if cap.isOpened():
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap, f"{protocol} Connected", candidate, protocol
                cap.release()

            # Fallback to CAP_ANY
            cap = cv2.VideoCapture(candidate)
            if cap.isOpened():
                success, frame = cap.read()
                if success and frame is not None and frame.size > 0:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return cap, f"{protocol} Connected", candidate, protocol
                cap.release()
        except Exception as exc:
            logger.debug("Candidate stream %s open failed: %s", candidate, exc)

    return None, f"Could not open feed from {canonical_url} (checked {len(candidates)} endpoints)", canonical_url, protocol


def draw_tactical_hud(
    frame: np.ndarray,
    rtsp_url: str,
    fps: float,
    detections_count: int,
    is_live: bool = True,
    protocol: str = "LIVE FEED",
    breach_label: Optional[str] = None,
) -> None:
    """Draw tactical border surveillance HUD on top of live video."""
    h, w = frame.shape[:2]

    # Top left tactical status tag
    cv2.rectangle(frame, (10, 10), (270, 42), (10, 15, 20), -1)
    cv2.rectangle(frame, (10, 10), (270, 42), (40, 60, 80), 1)

    dot_color = (0, 220, 120) if is_live else (0, 180, 255)
    cv2.circle(frame, (25, 26), 5, dot_color, -1)

    # Dynamic status text based on protocol
    short_proto = (
        "WEBCAM"
        if "Webcam" in protocol and "IP Webcam" not in protocol
        else ("IP-WEBCAM" if "IP Webcam" in protocol else ("RTSP" if "RTSP" in protocol else "FEED"))
    )
    status_text = f"{short_proto} LIVE" if is_live else "FEED SYNCING"
    cv2.putText(frame, status_text, (38, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"{fps:.1f} FPS", (195, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1)

    # Active Geofence Breach Banner in Top Center
    if breach_label:
        banner_text = f"🚨 BREACH DETECTED: {breach_label.upper()}"
        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cx = w // 2
        cv2.rectangle(frame, (cx - tw // 2 - 16, 8), (cx + tw // 2 + 16, 44), (20, 20, 220), -1)
        cv2.rectangle(frame, (cx - tw // 2 - 16, 8), (cx + tw // 2 + 16, 44), (255, 255, 255), 2)
        cv2.putText(frame, banner_text, (cx - tw // 2, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # Top right timestamp
    now_str = datetime.now().strftime("%H:%M:%S")
    cv2.rectangle(frame, (w - 180, 10), (w - 10, 42), (10, 15, 20), -1)
    cv2.rectangle(frame, (w - 180, 10), (w - 10, 42), (40, 60, 80), 1)
    cv2.putText(frame, f"BOP-04 • {now_str}", (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 120), 1)

    # Bottom detection summary
    if detections_count > 0:
        cv2.rectangle(frame, (10, h - 35), (260, h - 10), (15, 20, 40), -1)
        cv2.rectangle(frame, (10, h - 35), (260, h - 10), (60, 80, 220), 1)
        cv2.putText(
            frame,
            f"TARGETS ACQUIRED: {detections_count}",
            (18, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (50, 100, 255),
            1,
        )


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: list[dict],
) -> None:
    """Draw tactical AI threat detection boxes and labels on frame."""
    h, w = frame.shape[:2]
    for det in detections:
        box = det.get("box", [])
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        cls_name = det.get("class_name", "Object").lower()
        conf = det.get("confidence", 0.0)
        is_breach = bool(det.get("is_breach", False))
        breach_label = det.get("breach_label", "PERIMETER BREACH")

        color = (0, 0, 255) if is_breach else CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        box_thickness = 3 if is_breach else 2

        # Draw main box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

        # Draw corner accents
        corner_len = min(15, (x2 - x1) // 3, (y2 - y1) // 3)
        if corner_len > 3:
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

        # Highlight ground footpoint with bullseye on breach
        if is_breach:
            fpx = (x1 + x2) // 2
            fpy = y2
            cv2.circle(frame, (fpx, fpy), 7, (0, 0, 255), -1)
            cv2.circle(frame, (fpx, fpy), 13, (0, 255, 255), 2)
            cv2.line(frame, (fpx - 18, fpy), (fpx + 18, fpy), (0, 255, 255), 1)
            cv2.line(frame, (fpx, fpy - 18), (fpx, fpy + 18), (0, 255, 255), 1)

        # Label background pill
        global_trace_id = det.get("global_trace_id")
        is_cross_camera = bool(det.get("is_cross_camera", False))
        track_id = det.get("track_id")
        if is_breach:
            label = f"🚨 BREACH: {breach_label.upper()} [#{track_id}]"
        elif global_trace_id:
            cross_icon = " ⇄" if is_cross_camera else ""
            id_prefix = f"{global_trace_id}{cross_icon} | "
            label = f"{id_prefix}{cls_name.upper()} {int(conf * 100)}%"
        elif track_id is not None:
            id_prefix = f"#{track_id} "
            label = f"{id_prefix}{cls_name.upper()} {int(conf * 100)}%"
        else:
            label = f"{cls_name.upper()} {int(conf * 100)}%"

        (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        pill_y1 = max(0, y1 - lbl_h - 6)
        pill_y2 = y1
        pill_color = (0, 0, 220) if is_breach else color
        text_color = (255, 255, 255) if is_breach else (0, 0, 0)
        cv2.rectangle(frame, (x1, pill_y1), (x1 + lbl_w + 8, pill_y2), pill_color, -1)
        cv2.rectangle(frame, (x1, pill_y1), (x1 + lbl_w + 8, pill_y2), (255, 255, 255), 1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, pill_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            text_color,
            1,
            cv2.LINE_AA,
        )


def stream_generator(
    rtsp_url: str,
    draw_detections: bool = True,
    draw_hud: bool = False,
    conf_threshold: float = 0.30,
    model_name: str = "best",
    fps_limit: int = 24,
    rotation: int = 0,
    enable_face_recognition: bool = False,
    camera_id: Optional[str] = None,
    camera_name: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
    session_id: Optional[str] = None,
) -> Generator[bytes, None, None]:
    """
    Generator yielding multipart JPEG frames from Local Webcam, RTSP, or IP Webcam Pro stream.
    Gracefully falls back to tactical standby or demo stream when link is offline.

    Args:
        stop_event: Optional threading.Event. When set (via DELETE /api/stream/live/stop),
                    the generator exits within one frame interval, stopping all inference.
        session_id: Client session ID used to publish real-time structured detection telemetry.
    """
    frame_interval = 1.0 / max(1, min(fps_limit, 30))
    cap, conn_diag, resolved_url, protocol = open_video_source(rtsp_url)
    clean_url = rtsp_url.strip().lower()
    is_video_file = (
        clean_url in ("sample", "demo", "test")
        or not clean_url
        or Path(rtsp_url).is_file()
    )

    effective_cam_id = camera_id or resolved_url or rtsp_url
    effective_cam_name = camera_name or protocol or "Live Stream"

    if draw_detections:
        enable_face_recognition = True

    # Initialize AI preprocessor whenever detection model is loaded
    preprocessor = None
    if model_manager.is_loaded(model_name):
        try:
            preprocessor = Preprocessor(
                PreprocessingConfig(
                    target_width=640,
                    target_height=640,
                    convert_bgr_to_rgb=True,
                    normalize=False,
                    scale=1.0 / 255.0,
                    channel_first=True,
                    add_batch_dimension=True,
                )
            )
        except Exception as exc:
            logger.warning("Could not create preprocessor for live stream: %s", exc)

    cached_detections: list[dict] = []
    cached_faces: list[dict] = []
    recent_track_breaches: dict[int, tuple[str, str, float]] = {}  # track_id -> (name, type, timestamp)
    face_id_track_map: dict[str, int] = {}  # face identity -> stable synthetic track_id for geofence
    _face_id_counter: list[int] = [0]  # mutable counter via list
    active_breach_name: Optional[str] = None
    active_breach_time: float = 0.0
    frame_count = 0
    last_fps_time = time.perf_counter()
    measured_fps = float(fps_limit)

    reconnect_attempts = 0
    max_reconnects = 5

    try:
        while True:
            # ── Client stop-signal check ──────────────────────────────────────
            # The frontend calls DELETE /api/stream/live/stop?session_id=X when
            # the user navigates away.  This exits the loop within one frame
            # interval, stopping all YOLO / face-recognition inference.
            if stop_event is not None and stop_event.is_set():
                logger.info("Live stream: stop signal received, shutting down (%s)", rtsp_url)
                break

            t_start = time.perf_counter()

            if cap is None or not cap.isOpened():
                # Attempt periodic reconnect
                reconnect_attempts += 1
                if reconnect_attempts <= max_reconnects:
                    time.sleep(0.5)
                    cap, conn_diag, resolved_url, protocol = open_video_source(rtsp_url)
                    if cap and cap.isOpened():
                        reconnect_attempts = 0
                        continue

                # Stream standby frame while waiting for link
                status_label = (
                    f"Connecting to {protocol} ({conn_diag})..."
                    if reconnect_attempts < 3
                    else f"{protocol} Offline • {conn_diag}"
                )
                standby = create_standby_frame(
                    rtsp_url=rtsp_url,
                    status_msg=status_label,
                    protocol=protocol,
                )
                ret, jpeg = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )
                time.sleep(0.5)
                # Attempt periodic reconnect every few iterations
                cap, conn_diag, resolved_url, protocol = open_video_source(rtsp_url)
                continue

            # Flush buffered frame backlog for live RTSP to prevent latency accumulation
            if not is_video_file and cap is not None:
                cap.grab()

            success, frame = cap.read()

            if not success or frame is None:
                if is_video_file and cap is not None:
                    # Loop local sample video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Stream read hiccup
                reconnect_attempts += 1
                if reconnect_attempts > 4:
                    cap.release()
                    cap = None
                time.sleep(0.1)
                continue

            reconnect_attempts = 0
            frame_count += 1

            # Update FPS measurement every 15 frames
            if frame_count % 15 == 0:
                elapsed = time.perf_counter() - last_fps_time
                if elapsed > 0:
                    measured_fps = 15.0 / elapsed
                last_fps_time = time.perf_counter()

            # ── Staggered AI Pipeline (Zero-Stutter Cadence) ─────────────────────
            # Frame cadence across every 4 frames:
            #   Frame % 4 == 0 -> (reserved - threat detection removed)
            #   Frame % 4 == 1 -> Fast pass-through (~5 ms)
            #   Frame % 4 == 2 -> Biometric Face Detection (~35 ms)
            #   Frame % 4 == 3 -> Fast pass-through (~5 ms)

            # Stage 2: Biometric Facial Recognition (with Distant Person-RoI & Multi-Scale Scanning)
            if enable_face_recognition:
                if frame_count % 4 == 2 or not cached_faces:
                    try:
                        from app.pipeline.face_service import face_service
                        person_boxes = [
                            d.get("box") for d in cached_detections
                            if (str(d.get("class_name", "")).lower() in ("person", "human", "pedestrian") or d.get("class_id") == 0)
                            and d.get("box")
                        ]
                        cached_faces = face_service.detect_and_recognize(
                            frame,
                            min_match_score=None,
                            min_face_size=12,
                            det_score_thresh=0.25,
                            use_temporal_smoothing=True,
                            max_det_width=1280,
                            enable_distant_mode=True,
                            person_boxes=person_boxes if person_boxes else None,
                        )
                        if cached_faces:
                            try:
                                from app.pipeline.unknown_face_service import unknown_face_service
                                for fc in cached_faces:
                                    if not fc.get("is_known"):
                                        unknown_face_service.record_unknown(
                                            frame=frame,
                                            face=fc,
                                            camera_id=effective_cam_id,
                                            camera_name=effective_cam_name,
                                        )
                            except Exception as unk_err:
                                logger.debug("Live stream unknown record exception: %s", unk_err)

                        if cached_faces and cached_detections:
                            try:
                                from app.tracking.global_tracker import match_face_to_person_track, global_trace_manager
                                for fc in cached_faces:
                                    if fc.get("is_threat") or fc.get("is_known"):
                                        fc_box = fc.get("bbox") or fc.get("box")
                                        if fc_box:
                                            mp = match_face_to_person_track(fc_box, cached_detections)
                                            if mp:
                                                f_name = fc.get("name", "Unknown")
                                                is_th = bool(fc.get("is_threat", False))
                                                t_lvl = fc.get("threat_level") or ("HIGH" if is_th else "LOW")
                                                cat = fc.get("category") or ("SUSPECT" if is_th else "PERSONNEL")
                                                f_conf = float(fc.get("calibrated_conf") or fc.get("match_score") or fc.get("confidence") or 0.85)
                                                mp["suspect_name"] = f_name
                                                mp["is_threat"] = is_th
                                                mp["threat_level"] = t_lvl
                                                mp["category"] = cat
                                                mp["biometric_locked"] = True
                                                gtid = mp.get("global_trace_id")
                                                if gtid:
                                                    global_trace_manager.resolve_trace_suspect(
                                                        global_track_id=gtid,
                                                        suspect_name=f_name,
                                                        threat_level=t_lvl,
                                                        category=cat,
                                                        face_confidence=f_conf,
                                                        source_camera_id=effective_cam_id,
                                                    )
                            except Exception as fusion_err:
                                logger.debug("Live stream face-to-track fusion error: %s", fusion_err)
                    except Exception as exc:
                        logger.debug("Live stream face recognition exception: %s", exc)

            # Stage 3: Geofence & Directional Tripwire Evaluation (face-based)
            # Runs on every face-recognition frame (frame_count % 4 == 2) or when faces change.
            # Uses cached_faces converted to geofence-compatible objects with stable synthetic track IDs.
            if cached_faces:
                try:
                    from app.pipeline.geofence_engine import geofence_engine
                    h_fr, w_fr = frame.shape[:2]
                    geofence_objects: list[dict] = []
                    for fc in cached_faces:
                        fc_box = fc.get("bbox") or fc.get("box")
                        if not fc_box or len(fc_box) < 4:
                            continue
                        fx1, fy1, fx2, fy2 = [float(v) for v in fc_box[:4]]
                        face_h = max(10.0, fy2 - fy1)
                        face_w = max(10.0, fx2 - fx1)
                        # Upright person ground contact estimation:
                        # Full height ~7.2x face height, with feet resting on the ground
                        est_feet_y = min(float(h_fr), fy1 + face_h * 7.2)
                        est_body_x1 = max(0.0, fx1 - face_w * 0.8)
                        est_body_x2 = min(float(w_fr), fx2 + face_w * 0.8)
                        person_box = [est_body_x1, fy1, est_body_x2, est_feet_y]

                        # Build a stable synthetic track_id keyed on face identity/id string
                        face_key = str(fc.get("id") or fc.get("name") or fc.get("face_id") or str(fc_box))
                        if face_key not in face_id_track_map:
                            _face_id_counter[0] += 1
                            face_id_track_map[face_key] = _face_id_counter[0]
                        synthetic_tid = face_id_track_map[face_key]
                        f_name = fc.get("name", "Unknown")
                        is_th = bool(fc.get("is_threat", False))
                        geofence_objects.append({
                            "box": person_box,
                            "face_box": fc_box,
                            "head_point": [(fx1 + fx2) / (2.0 * max(1, w_fr)), fy2 / float(max(1, h_fr))],
                            "track_id": synthetic_tid,
                            "class_name": "suspect" if is_th else "person",
                            "confidence": float(fc.get("calibrated_conf") or fc.get("match_score") or fc.get("confidence") or 0.85),
                            "suspect_name": f_name if f_name != "Unknown" else None,
                        })
                    if geofence_objects:
                        breaches = geofence_engine.evaluate_tracks(
                            camera_id=effective_cam_id,
                            tracked_objects=geofence_objects,
                            frame_resolution=(w_fr, h_fr),
                            auto_alert=True,
                            frame=frame,
                            camera_name=effective_cam_name,
                        )
                        now_t = time.time()
                        if breaches:
                            for b in breaches:
                                bid = b.get("trackId")
                                if bid is not None:
                                    b_name = b.get("name", "PERIMETER")
                                    b_type = b.get("type", "zone_breach")
                                    recent_track_breaches[bid] = (b_name, b_type, now_t)
                                    active_breach_name = b_name
                                    active_breach_time = now_t
                        # Apply breach highlights to matching faces in cached_faces
                        for geo_obj in geofence_objects:
                            tid = geo_obj["track_id"]
                            if tid in recent_track_breaches:
                                b_name, b_type, b_time = recent_track_breaches[tid]
                                if now_t - b_time < 3.5:
                                    for fc in cached_faces:
                                        fc_b = fc.get("bbox") or fc.get("box")
                                        f_key = str(fc.get("id") or fc.get("name") or fc.get("face_id") or str(fc_b))
                                        if face_id_track_map.get(f_key) == tid:
                                            fc["is_breach"] = True
                                            fc["breach_label"] = b_name
                                            fc["breach_type"] = b_type
                except Exception as geo_err:
                    logger.debug("Geofence face-based evaluation skip: %s", geo_err)

            # Draw Biometric Face Overlays (from cached faces)
            if (enable_face_recognition or draw_detections) and cached_faces:
                try:
                    from app.pipeline.face_service import face_service
                    frame = face_service.draw_faces(frame, cached_faces)
                except Exception as draw_exc:
                    logger.debug("Live stream face draw exception: %s", draw_exc)

            # Draw Active Geofence Breach Banner in Top Center
            now_curr = time.time()
            if active_breach_name and (now_curr - active_breach_time < 4.0):
                h_b, w_b = frame.shape[:2]
                banner = f"🚨 PERIMETER BREACH DETECTED: {active_breach_name.upper()} 🚨"
                (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                bx1 = max(0, (w_b - bw) // 2 - 14)
                bx2 = min(w_b, (w_b + bw) // 2 + 14)
                cv2.rectangle(frame, (bx1, 10), (bx2, 10 + bh + 14), (0, 0, 200), -1)
                cv2.rectangle(frame, (bx1, 10), (bx2, 10 + bh + 14), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    banner,
                    ((w_b - bw) // 2, 10 + bh + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if draw_hud:
                total_det_count = len(cached_detections) + (len(cached_vehicles) if enable_vehicle_detection else 0)
                draw_tactical_hud(
                    frame,
                    rtsp_url=resolved_url,
                    fps=measured_fps,
                    detections_count=total_det_count,
                    is_live=not is_video_file,
                    protocol=protocol,
                    breach_label=active_breach_name if (time.time() - active_breach_time < 4.0) else None,
                )

            # Update real-time detections registry for active session
            if session_id:
                formatted_dets: list[dict] = []
                suspects_in_stream: list[dict] = []

                for d in cached_detections:
                    bx = d.get("box", [])
                    if len(bx) >= 4:
                        c_name = str(d.get("class_name", "object")).lower()
                        c_conf = float(d.get("confidence", 0.0))
                        t_id = d.get("track_id")
                        is_br = bool(d.get("is_breach", False))
                        b_lbl = d.get("breach_label")
                        gtid = d.get("global_trace_id")
                        is_th = c_name in ("firearm", "explosive", "melee_weapon", "blunt_weapon") or is_br
                        t_lvl = "CRITICAL" if is_br else ("HIGH" if is_th else None)
                        formatted_dets.append({
                            "box": [int(v) for v in bx],
                            "class_id": d.get("class_id", 0),
                            "class_name": c_name,
                            "confidence": round(c_conf, 3),
                            "track_id": t_id,
                            "is_threat": is_th,
                            "threat_level": t_lvl,
                            "is_breach": is_br,
                            "breach_label": b_lbl,
                            "global_trace_id": gtid,
                        })

                if enable_face_recognition and cached_faces:
                    for fc in cached_faces:
                        fc_box = fc.get("bbox") or fc.get("box")
                        if fc_box and len(fc_box) >= 4:
                            f_name = fc.get("name", "Unknown")
                            is_th = bool(fc.get("is_threat", False))
                            f_conf = float(fc.get("calibrated_conf") or fc.get("match_score") or fc.get("confidence") or 0.85)
                            t_lvl = fc.get("threat_level", "HIGH" if is_th else None)
                            cat = fc.get("category", "Suspect Target" if is_th else "Personnel")
                            formatted_dets.append({
                                "box": [int(v) for v in fc_box],
                                "class_id": 999,
                                "class_name": f"suspect_{f_name.lower()}" if is_th else f"face_{f_name.lower()}",
                                "confidence": round(f_conf, 3),
                                "is_threat": is_th,
                                "threat_level": t_lvl,
                                "suspect_name": f_name,
                                "category": cat,
                            })
                            if is_th:
                                suspects_in_stream.append({
                                    "name": f_name,
                                    "threat_level": t_lvl or "HIGH",
                                    "confidence": round(f_conf, 3),
                                    "category": cat,
                                })



                _live_session_detections[session_id] = {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "fps": round(measured_fps, 1),
                    "latency_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    "detections": formatted_dets,
                    "suspects": suspects_in_stream,
                    "count": len(formatted_dets),
                }

            # Encode frame to JPEG (75% quality for fast compression and high visual fidelity)
            ret, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )

            # Enforce target FPS rate
            t_work = time.perf_counter() - t_start
            sleep_time = frame_interval - t_work
            if sleep_time > 0:
                time.sleep(sleep_time)

    except GeneratorExit:
        logger.info("Client closed live stream connection: %s", rtsp_url)
    finally:
        if cap is not None:
            cap.release()
        if session_id:
            _live_session_detections.pop(session_id, None)


@router.get("/validate")
def validate_stream(
    stream_url: str = Query(
        ...,
        description="Stream URL (RTSP, HTTP IP Webcam, local webcam '0', or 'sample')",
    ),
) -> dict:
    """
    Pre-flight validation of stream reachability, protocol detection,
    and endpoint candidate resolution before connecting.
    """
    canonical_url, candidates, protocol = normalize_stream_url(stream_url)
    if canonical_url == "sample":
        return {
            "reachable": True,
            "status": "ready",
            "protocol": protocol,
            "target_url": "sample",
            "resolved_url": "sample",
            "latency_ms": 1,
            "message": "Sample demonstration video ready on server.",
            "is_ip_webcam": False,
        }

    reachable, diag, host, port = is_host_reachable(canonical_url, timeout_seconds=1.5)
    is_ip_webcam = "IP Webcam" in protocol

    return {
        "reachable": reachable,
        "status": "online" if reachable else "unreachable",
        "protocol": protocol,
        "target_url": stream_url,
        "resolved_url": candidates[0] if candidates else canonical_url,
        "host": host,
        "port": port,
        "latency_ms": 1 if ("Webcam" in protocol and reachable) else (25 if reachable else 0),
        "message": diag,
        "is_ip_webcam": is_ip_webcam,
        "candidates": candidates,
    }


@router.get("/live/detections")
def get_live_detections(
    session_id: str = Query(..., description="Active stream session ID"),
) -> dict:
    """Get the latest real-time AI detections for an active live stream session."""
    data = _live_session_detections.get(session_id)
    if data:
        # Check staleness: if no update in 4 seconds, mark stale and clear targets
        if time.time() - data.get("timestamp", 0) > 4.0:
            return {
                "session_id": session_id,
                "status": "stale",
                "timestamp": data.get("timestamp", 0),
                "fps": 0,
                "latency_ms": 0,
                "detections": [],
                "suspects": [],
                "count": 0,
            }
        return {
            "session_id": session_id,
            "status": "active",
            **data,
        }
    return {
        "session_id": session_id,
        "status": "idle",
        "timestamp": time.time(),
        "fps": 0,
        "latency_ms": 0,
        "detections": [],
        "suspects": [],
        "count": 0,
    }


@router.delete("/live/stop")
def stop_live_stream(
    session_id: str = Query(..., description="Stream session ID to stop"),
) -> dict:
    """Signal the live surveillance stream generator to stop immediately.

    The frontend calls this when the user navigates away from Live Surveillance.
    The generator exits its while-True loop within one frame interval, stopping
    all YOLO / face-recognition inference and releasing GPU resources.
    """
    _live_session_detections.pop(session_id, None)
    event = _live_stop_events.pop(session_id, None)
    if event is not None:
        event.set()
        logger.info("Live stream stop signal sent for session: %s", session_id)
        return {"status": "stopping", "session_id": session_id}
    logger.debug(
        "Live stream stop: session %s not found (may have already ended)", session_id
    )
    return {"status": "not_found", "session_id": session_id}


@router.get("/live")
def get_live_stream(
    rtsp_url: str = Query(
        ...,
        description="RTSP URL (e.g. rtsp://10.20.72.101:554/live), HTTP IP Webcam (http://ip:8080), or 'sample'",
    ),
    draw_detections: bool = Query(
        False,
        description="Overlay real-time ONNX threat detection bounding boxes",
    ),
    draw_hud: bool = Query(
        False,
        description="Overlay tactical OpenCV HUD text (False recommended for lowest latency)",
    ),
    conf_threshold: float = Query(
        0.30,
        ge=0.05,
        le=0.95,
        description="Minimum confidence threshold for detections",
    ),
    model_name: str = Query(
        "best",
        description="Loaded ONNX model name to use for detections",
    ),
    fps: int = Query(
        25,
        ge=5,
        le=60,
        description="Target stream framerate",
    ),
    enable_face_recognition: bool = Query(
        False,
        description="Overlay real-time facial recognition and identity labels",
    ),
    camera_id: Optional[str] = Query(
        None,
        description="Optional camera UUID / ID for multi-camera Re-ID tracking",
    ),
    camera_name: Optional[str] = Query(
        None,
        description="Optional friendly camera name (e.g. North Gate)",
    ),
    session_id: Optional[str] = Query(
        None,
        description="Unique client session ID.  When provided, DELETE /api/stream/live/stop stops inference.",
    ),
) -> StreamingResponse:
    """
    Stream live RTSP or IP Webcam CCTV camera feed as multipart/x-mixed-replace (MJPEG)
    for native, zero-plugin display in any web browser.
    """
    if draw_detections:
        enable_face_recognition = True

    sid = session_id or str(uuid.uuid4())
    stop_event = threading.Event()
    _live_stop_events[sid] = stop_event

    def _gen_with_cleanup():
        try:
            yield from stream_generator(
                rtsp_url=rtsp_url,
                draw_detections=draw_detections,
                draw_hud=draw_hud,
                conf_threshold=conf_threshold,
                model_name=model_name,
                fps_limit=fps,
                enable_face_recognition=enable_face_recognition,
                camera_id=camera_id,
                camera_name=camera_name,
                stop_event=stop_event,
                session_id=sid,
            )
        finally:
            _live_stop_events.pop(sid, None)
            _live_session_detections.pop(sid, None)

    return StreamingResponse(
        _gen_with_cleanup(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


@router.get("/snapshot")
def get_snapshot(
    rtsp_url: str = Query(..., description="Stream URL (RTSP or IP Webcam) or 'sample'"),
    draw_detections: bool = Query(False),
) -> Response:
    """Capture a single frame snapshot from an RTSP or IP Webcam camera stream."""
    try:
        cap, diag, resolved_url, protocol = open_video_source(rtsp_url)
    except Exception as exc:
        logger.warning("open_video_source failed for snapshot %s: %s", rtsp_url, exc)
        cap, diag, resolved_url, protocol = None, str(exc), rtsp_url, "Stream"

    frame = None

    if cap and cap.isOpened():
        try:
            success, img = cap.read()
            cap.release()
            if success and img is not None:
                frame = img
        except Exception as exc:
            logger.warning("Snapshot read frame failed: %s", exc)
            if cap:
                cap.release()

    if frame is None:
        frame = create_standby_frame(rtsp_url, status_msg=f"Snapshot: {diag}", protocol=protocol)

    if draw_detections and model_manager.is_loaded("best"):
        try:
            preprocessor = Preprocessor(
                PreprocessingConfig(target_width=640, target_height=640)
            )
            tensor = preprocessor.process(frame)
            h, w = frame.shape[:2]
            res = inference_service.predict(
                model_name="best",
                input_data=tensor,
                postprocess=True,
                conf_threshold=0.30,
                iou_threshold=0.45,
                original_image_size=(w, h),
            )
            draw_bounding_boxes(frame, res.get("detections", []))
        except Exception as exc:
            logger.warning("Snapshot detection failed: %s", exc)

    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ret:
        fallback = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", fallback)

    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )
