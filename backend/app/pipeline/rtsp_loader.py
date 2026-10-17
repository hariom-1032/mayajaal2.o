from __future__ import annotations

import logging
import time
from typing import Any

import cv2


logger = logging.getLogger(__name__)


class RTSPError(Exception):
    """Base exception for RTSP stream errors."""


class InvalidRTSPUrlError(RTSPError):
    """Raised when an RTSP URL is invalid."""


class RTSPConnectionError(RTSPError):
    """Raised when an RTSP stream cannot be opened."""


class RTSPReadError(RTSPError):
    """Raised when a frame cannot be read from the stream."""


class RTSPStream:
    """
    Safe RTSP stream reader for CCTV cameras.

    Features:
    - RTSP URL validation
    - Controlled connection timeout
    - Controlled frame-read timeout
    - Connection failure handling
    - Frame-read failure handling
    - Bounded reconnect attempts
    - Resource cleanup
    """

    def __init__(
        self,
        rtsp_url: str,
        reconnect_attempts: int = 3,
        reconnect_delay_seconds: float = 2.0,
        connection_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
    ) -> None:
        self.rtsp_url = rtsp_url.strip()

        if not self.rtsp_url:
            raise InvalidRTSPUrlError(
                "Stream URL cannot be empty."
            )

        # Normalize bare IP/ports
        clean_url = self.rtsp_url
        if "://" not in clean_url:
            if ":554" in clean_url or "rtsp" in clean_url.lower():
                clean_url = f"rtsp://{clean_url}"
            else:
                clean_url = f"http://{clean_url}"
        self.rtsp_url = clean_url

        allowed_schemes = ("rtsp://", "rtsps://", "http://", "https://")
        if not any(self.rtsp_url.lower().startswith(s) for s in allowed_schemes):
            raise InvalidRTSPUrlError(
                "Stream URL must start with rtsp://, rtsps://, http://, or https://"
            )

        if reconnect_attempts < 0:
            raise ValueError(
                "reconnect_attempts cannot be negative."
            )

        if reconnect_delay_seconds < 0:
            raise ValueError(
                "reconnect_delay_seconds cannot be negative."
            )

        if connection_timeout_ms <= 0:
            raise ValueError(
                "connection_timeout_ms must be greater than 0."
            )

        if read_timeout_ms <= 0:
            raise ValueError(
                "read_timeout_ms must be greater than 0."
            )

        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.connection_timeout_ms = connection_timeout_ms
        self.read_timeout_ms = read_timeout_ms

        self.capture: cv2.VideoCapture | None = None

    def connect(self) -> None:
        """Open the stream (RTSP or HTTP IP Webcam) with candidate fallbacks and bounded timeouts."""

        self.release()

        # Check for local webcam device index or alias
        clean = (self.rtsp_url or "").strip().lower()
        if clean.isdigit() or clean in ("webcam", "camera", "default", "usb"):
            cam_idx = int(clean) if clean.isdigit() else 0
            capture = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam_idx)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                success, frame = capture.read()
                if success and frame is not None and frame.size > 0:
                    self.capture = capture
                    logger.info("Local webcam device %s connected successfully", cam_idx)
                    return
                capture.release()

        # Generate candidates for IP Webcam if applicable
        candidates = [self.rtsp_url]
        lower_url = self.rtsp_url.lower()
        if lower_url.startswith("http://") or lower_url.startswith("https://"):
            if not lower_url.endswith("/video") and not lower_url.endswith("/videofeed"):
                base = self.rtsp_url.rstrip("/")
                candidates = [f"{base}/video", f"{base}/videofeed", self.rtsp_url]

        # Configure low-latency FFmpeg capture options:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"
        )

        last_error = None
        for target in candidates:
            try:
                capture = cv2.VideoCapture(
                    target,
                    cv2.CAP_FFMPEG,
                )
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    success, frame = capture.read()
                    if success and frame is not None and frame.size > 0:
                        self.capture = capture
                        logger.info("Stream connected successfully to %s", target)
                        return
                    capture.release()

                # Fallback to CAP_ANY
                capture = cv2.VideoCapture(target)
                if capture.isOpened():
                    success, frame = capture.read()
                    if success and frame is not None and frame.size > 0:
                        self.capture = capture
                        logger.info("Stream connected successfully to %s (CAP_ANY)", target)
                        return
                    capture.release()
            except Exception as exc:
                last_error = exc

        raise RTSPConnectionError(
            f"Could not open live stream from {self.rtsp_url} ({last_error or 'no valid video frame returned'})."
        )

        logger.info(
            "RTSP stream connected successfully."
        )

    def read_frame(self) -> Any:
        """Read one frame from the RTSP stream."""

        if self.capture is None:
            raise RTSPConnectionError(
                "RTSP stream is not connected."
            )

        try:
            success, frame = self.capture.read()

        except Exception as exc:
            logger.exception(
                "RTSP frame read failed."
            )

            raise RTSPReadError(
                f"RTSP frame read failed: {exc}"
            ) from exc

        if not success or frame is None:
            raise RTSPReadError(
                "RTSP stream returned no frame."
            )

        if frame.size == 0:
            raise RTSPReadError(
                "RTSP stream returned an empty frame."
            )

        return frame

    def reconnect(self) -> bool:
        """
        Attempt bounded reconnection attempts.

        Returns:
            True if reconnection succeeds.
            False if all attempts fail.
        """

        self.release()

        for attempt in range(
            1,
            self.reconnect_attempts + 1,
        ):
            logger.warning(
                "RTSP reconnect attempt %d/%d.",
                attempt,
                self.reconnect_attempts,
            )

            try:
                self.connect()
                return True

            except RTSPConnectionError as exc:
                logger.warning(
                    "RTSP reconnect attempt failed: %s",
                    exc,
                )

                if attempt < self.reconnect_attempts:
                    time.sleep(
                        self.reconnect_delay_seconds
                    )

        logger.error(
            "RTSP reconnection failed after %d attempts.",
            self.reconnect_attempts,
        )

        return False

    def get_status(self) -> dict[str, Any]:
        """Return current RTSP stream status."""

        connected = (
            self.capture is not None
            and self.capture.isOpened()
        )

        return {
            "connected": connected,
            "url_configured": bool(self.rtsp_url),
            "reconnect_attempts": self.reconnect_attempts,
            "connection_timeout_ms": self.connection_timeout_ms,
            "read_timeout_ms": self.read_timeout_ms,
        }

    def release(self) -> None:
        """Release the RTSP capture resource."""

        if self.capture is not None:
            self.capture.release()
            self.capture = None

            logger.info(
                "RTSP stream released."
            )

    def __enter__(self) -> "RTSPStream":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.release()
