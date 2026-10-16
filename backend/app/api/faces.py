from __future__ import annotations

import base64
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Generator, Optional

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel

from app.core.config import settings
from app.pipeline.face_service import face_service

logger = logging.getLogger("SIH26187.FaceAPI")

router = APIRouter(
    prefix="/api/faces",
    tags=["Facial Recognition"],
)

SAMPLE_VIDEO_PATH = settings.model_directory.parent / "test_video_input.mp4"


def decode_image_input(
    file_bytes: Optional[bytes] = None,
    base64_str: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Decode raw image bytes or base64 data into a BGR numpy array."""
    try:
        if file_bytes:
            nparr = np.frombuffer(file_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if base64_str:
            clean_b64 = base64_str
            if "base64," in clean_b64:
                clean_b64 = clean_b64.split("base64,")[1]
            image_data = base64.b64decode(clean_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("Failed to decode image input: %s", exc)
    return None


@router.get("")
def list_enrolled_faces() -> dict:
    """Retrieve all enrolled personnel in the MAATRIX Face Recognition database."""
    persons = face_service.list_persons()
    return {
        "success": True,
        "count": len(persons),
        "persons": persons,
    }


@router.get("/status")
def get_face_engine_status() -> dict:
    """Get operational status of YuNet and SFace models and enrolled personnel count."""
    return {
        "status": "ready" if (face_service.detector and face_service.recognizer) else "degraded",
        "detector_loaded": face_service.detector is not None,
        "recognizer_loaded": face_service.recognizer is not None,
        "enrolled_count": len(face_service.known_persons),
        "yunet_model": str(face_service.yunet_path.name),
        "sface_model": str(face_service.sface_path.name),
    }


@router.get("/events")
def get_recent_face_events(limit: int = Query(30, ge=1, le=100)) -> dict:
    """Return the most recent facial detection and identification events for live telemetry."""
    events = face_service.recent_events[:limit]
    return {
        "success": True,
        "count": len(events),
        "events": events,
    }


class PersonMetadataUpdate(BaseModel):
    is_suspect: Optional[bool] = None
    threat_level: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


@router.post("/register")
async def register_face(
    name: str = Form(None),
    is_suspect: bool = Form(True),
    threat_level: str = Form("HIGH"),
    category: str = Form("Wanted / BOLO"),
    notes: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
) -> dict:
    """
    Enroll a new person or add an identity profile with name, face image, and suspect metadata.
    Accepts multipart file upload OR base64 data string.
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Name is required for face registration.")

    image_bgr: Optional[np.ndarray] = None

    if file:
        content = await file.read()
        image_bgr = decode_image_input(file_bytes=content)
    elif image_base64:
        image_bgr = decode_image_input(base64_str=image_base64)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="A valid face image file or base64 data is required.")

    result = face_service.register_face(
        image_bgr,
        name.strip(),
        is_suspect=is_suspect,
        threat_level=threat_level,
        category=category,
        notes=notes or "",
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to register face"))

    return result


@router.patch("/{person_id}")
def update_face_metadata(
    person_id: str,
    payload: PersonMetadataUpdate,
) -> dict:
    """Update suspect status, threat level, category, or notes for an enrolled identity."""
    res = face_service.update_person_metadata(
        person_id,
        is_suspect=payload.is_suspect,
        threat_level=payload.threat_level,
        category=payload.category,
        notes=payload.notes,
    )
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", "Failed to update metadata"))
    return res


@router.post("/add-sample")
async def add_face_sample(
    person_id: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
) -> dict:
    """Add an additional lighting or angle photo sample to an already enrolled identity."""
    image_bgr: Optional[np.ndarray] = None

    if file:
        content = await file.read()
        image_bgr = decode_image_input(file_bytes=content)
    elif image_base64:
        image_bgr = decode_image_input(base64_str=image_base64)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Valid image data is required.")

    result = face_service.add_sample(person_id, image_bgr)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to add sample"))

    return result


@router.delete("/{person_id}")
def delete_person(person_id: str) -> dict:
    """Remove a person from the enrolled face database."""
    res = face_service.delete_person(person_id)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail="Person ID not found")
    return res


@router.post("/scan")
async def scan_image(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    min_match_score: float = Form(0.40),
    min_face_size: int = Form(40),
) -> dict:
    """
    Perform face detection and identity recognition on a static image.
    Returns detected face boxes, identity names, match scores, and annotated visualization.
    """
    t0 = time.perf_counter()
    image_bgr: Optional[np.ndarray] = None

    if file:
        content = await file.read()
        image_bgr = decode_image_input(file_bytes=content)
    elif image_base64:
        image_bgr = decode_image_input(base64_str=image_base64)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Valid image is required.")

    faces = face_service.detect_and_recognize(
        image_bgr,
        min_match_score=min_match_score,
        min_face_size=min_face_size,
        use_temporal_smoothing=False,
    )

    annotated = face_service.draw_faces(image_bgr, faces)
    ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("utf-8") if ret else None

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    return {
        "success": True,
        "face_count": len(faces),
        "faces": faces,
        "latency_ms": latency_ms,
        "annotated_image": annotated_b64,
    }


@router.get("/thumbnail/{filename}")
def get_face_thumbnail(filename: str) -> Response:
    """Serve cropped aligned face thumbnail image."""
    safe_name = Path(filename).name
    file_path = face_service.thumbnails_dir / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(file_path, media_type="image/jpeg")


import asyncio
import threading
import uuid
from typing import AsyncGenerator, Generator, Optional

active_stream_caps: set[cv2.VideoCapture] = set()
stream_active_flags: dict[str, bool] = {}
stream_lock = threading.Lock()


def release_all_face_cameras() -> int:
    """Signal all running face stream generator loops to exit cleanly and release hardware webcams."""
    with stream_lock:
        active_count = len([v for v in stream_active_flags.values() if v])
        for sid in list(stream_active_flags.keys()):
            stream_active_flags[sid] = False
        return active_count


@router.post("/camera/stop")
def stop_face_camera() -> dict:
    """Immediately stop all active face camera streams and release hardware device locks."""
    released = release_all_face_cameras()
    logger.info("Explicitly stopped face camera streams. Released %d device(s).", released)
    return {"success": True, "released_cameras": released}


def _generate_face_stream(
    source: str = "sample",
    fps: int = 20,
) -> Generator[bytes, None, None]:
    """Generator yielding multipart MJPEG frames with live facial recognition HUD."""
    stream_id = str(uuid.uuid4())
    stream_active_flags[stream_id] = True
    cap = None
    consecutive_fails = 0

    try:
        # Determine capture source: local webcam index, RTSP URL, or sample video
        if source.isdigit():
            cam_idx = int(source)
            # Release any prior captures so the physical camera is unblocked
            release_all_face_cameras()
            time.sleep(0.12)
            stream_active_flags[stream_id] = True

            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if not cap or not cap.isOpened():
                cap = cv2.VideoCapture(cam_idx)
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif source == "sample" or not source:
            if SAMPLE_VIDEO_PATH.exists():
                cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
            else:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source)

        if not cap or not cap.isOpened():
            if source.isdigit():
                # Try fallback index 0 if specific index failed
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap or not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if cap and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap and cap.isOpened():
            with stream_lock:
                active_stream_caps.add(cap)

        frame_interval = 1.0 / max(5, min(30, fps))

        while True:
            # Check if this stream has been instructed to terminate
            if not stream_active_flags.get(stream_id, False):
                logger.info("Face stream %s terminated via active flag.", stream_id)
                break

            t0 = time.time()
            try:
                success, frame = cap.read() if cap and cap.isOpened() else (False, None)

                if not success or frame is None:
                    consecutive_fails += 1

                    # If sample video reached EOF, loop it
                    if (source == "sample" or not source) and SAMPLE_VIDEO_PATH.exists() and cap and cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_fails = 0
                        continue

                    # If camera failed or not opened, attempt reconnect every 20 loops (~2s)
                    if source.isdigit() and consecutive_fails >= 20:
                        consecutive_fails = 0
                        try:
                            if cap:
                                cap.release()
                            cam_idx = int(source)
                            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                            if not cap or not cap.isOpened():
                                cap = cv2.VideoCapture(cam_idx)
                            if cap and cap.isOpened():
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                with stream_lock:
                                    active_stream_caps.add(cap)
                                continue
                        except Exception as reconnect_err:
                            logger.debug("Camera reconnect attempt failed: %s", reconnect_err)

                    # Standby placeholder
                    standby = np.zeros((480, 640, 3), dtype=np.uint8)
                    standby[:] = (20, 24, 28)
                    status_text = "CAMERA FEED OFFLINE / CONNECTING..." if source.isdigit() else "FACIAL RECON | CONNECTING..."
                    cv2.putText(
                        standby,
                        status_text,
                        (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 230, 115),
                        2,
                    )
                    now_str = datetime.now().strftime("%H:%M:%S")
                    cv2.putText(
                        standby,
                        f"MAATRIX FACE HUD | {now_str} IST",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (160, 180, 200),
                        1,
                    )
                    ret, jpeg = cv2.imencode(".jpg", standby, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                        )
                    time.sleep(0.1)
                    continue

                consecutive_fails = 0

                # Downsample only if exceptionally high resolution (maintain up to 1280px for distant faces)
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280.0 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)))

                # Run Face detection & recognition with surveillance parameters and temporal smoothing
                try:
                    faces = face_service.detect_and_recognize(
                        frame,
                        min_match_score=0.34,
                        min_face_size=24,
                        det_score_thresh=0.45,
                        use_temporal_smoothing=True,
                    )
                    annotated = face_service.draw_faces(frame, faces)
                except Exception as model_err:
                    logger.warning("Face processing frame error: %s", model_err)
                    faces = []
                    annotated = frame

                # Stream HUD overlay
                now_str = datetime.now().strftime("%H:%M:%S")
                cv2.putText(
                    annotated,
                    f"MAATRIX FACE HUD | {now_str} IST",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 230, 115),
                    2,
                    cv2.LINE_AA,
                )
                count_label = f"DETECTED FACES: {len(faces)}"
                cv2.putText(
                    annotated,
                    count_label,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 220, 240),
                    1,
                    cv2.LINE_AA,
                )

                ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )

                elapsed = time.time() - t0
                sleep_time = max(0.01, frame_interval - elapsed)
                time.sleep(sleep_time)

            except GeneratorExit:
                break
            except Exception as loop_err:
                logger.warning("Stream loop hiccup: %s", loop_err)
                time.sleep(0.05)

    except GeneratorExit:
        pass
    except Exception as exc:
        logger.exception("Face stream generator error: %s", exc)
    finally:
        stream_active_flags.pop(stream_id, None)
        with stream_lock:
            if cap:
                active_stream_caps.discard(cap)
                try:
                    cap.release()
                    logger.info("Face camera device released successfully.")
                except Exception as exc:
                    logger.warning("Error releasing cap on exit: %s", exc)


@router.get("/stream")
def stream_faces(
    source: str = Query("sample", description="Camera index '0', RTSP URL, or 'sample'"),
    fps: int = Query(20, ge=5, le=30),
) -> StreamingResponse:
    """Live MJPEG video stream with real-time facial recognition and identity labeling."""
    return StreamingResponse(
        _generate_face_stream(source=source, fps=fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


@router.websocket("/ws/stream")
async def face_websocket_stream(websocket: WebSocket):
    """
    Bi-directional real-time WebSocket stream for client-side webcams.
    Receives video frame blobs from any remote browser (getUserMedia) across the network,
    runs YuNet face detection & SFace identity recognition on the backend server,
    and returns tactical annotated HUD frames to the client in real time.
    """
    await websocket.accept()
    logger.info("Face recognition WebSocket connected from %s", websocket.client)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None or frame.size == 0:
                    continue

                h, w = frame.shape[:2]
                if w > 854:
                    scale = 854.0 / w
                    frame = cv2.resize(frame, (854, int(h * scale)))

                # Run Face detection & recognition with temporal smoothing and surveillance thresholds
                try:
                    faces = face_service.detect_and_recognize(
                        frame,
                        min_match_score=0.34,
                        min_face_size=24,
                        det_score_thresh=0.45,
                        use_temporal_smoothing=True,
                    )
                    annotated = face_service.draw_faces(frame, faces)
                except Exception as model_err:
                    logger.warning("WebSocket face frame processing error: %s", model_err)
                    faces = []
                    annotated = frame

                # Stream HUD overlay
                now_str = datetime.now().strftime("%H:%M:%S")
                cv2.putText(
                    annotated,
                    f"MAATRIX FACE HUD | {now_str} IST",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 230, 115),
                    2,
                    cv2.LINE_AA,
                )
                count_label = f"DETECTED FACES: {len(faces)} | CLIENT WEBCAM"
                cv2.putText(
                    annotated,
                    count_label,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 220, 240),
                    1,
                    cv2.LINE_AA,
                )

                ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    await websocket.send_bytes(jpeg.tobytes())

            elif "text" in message and message["text"]:
                if message["text"] == "ping":
                    await websocket.send_text("pong")

    except WebSocketDisconnect:
        logger.info("Face recognition WebSocket disconnected.")
    except Exception as exc:
        logger.warning("Face recognition WebSocket error: %s", exc)
