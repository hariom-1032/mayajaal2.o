"""
/api/tracking — ByteTrack multi-object tracking endpoints.

Endpoints
---------
POST   /api/tracking/video
    Stateless video upload tracking (fresh tracker per request).

POST   /api/tracking/rtsp
    Stateful RTSP tracking using per-camera session (track IDs persist
    across multiple requests to the same camera_id).

GET    /api/tracking/rtsp/sessions
    List all active RTSP tracker sessions.

DELETE /api/tracking/rtsp/{camera_id}
    Reset and remove the tracker session for a specific camera.
"""
from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.core.runtime import (
    inference_service,
    model_manager,
)
from app.pipeline.preprocessor import (
    PreprocessingConfig,
    PreprocessingError,
    Preprocessor,
)
from app.pipeline.video_loader import VideoLoader, VideoLoaderError
from app.pipeline.rtsp_loader import (
    InvalidRTSPUrlError,
    RTSPConnectionError,
    RTSPError,
    RTSPReadError,
    RTSPStream,
)
from app.tracking.config import TrackerConfig
from app.tracking.tracker import ByteTrackerWrapper, ByteTrackerError
from app.tracking.session_store import tracker_session_store
from app.tracking.annotator import draw_tracked_boxes, convert_video_to_h264
from app.pipeline.face_service import face_service
from app.pipeline.alert_service import alert_service, SNAPSHOTS_DIR
from app.tracking.global_tracker import global_trace_manager


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/tracking",
    tags=["Tracking"],
)


# ---------------------------------------------------------------------------
# Helpers shared with /api/inference (local duplicates to avoid coupling)
# ---------------------------------------------------------------------------

def _get_model_input_size(model_name: str) -> tuple[int, int]:
    """Return preferred (width, height) for a model, defaulting to 640×640."""
    try:
        engine = model_manager.get_model(model_name)
        if getattr(engine, "input_size", None):
            return engine.input_size
    except Exception:
        pass
    return (640, 640)


def _create_preprocessor(target_width: int = 640, target_height: int = 640) -> Preprocessor:
    return Preprocessor(
        PreprocessingConfig(
            target_width=target_width,
            target_height=target_height,
            convert_bgr_to_rgb=True,
            normalize=False,
            scale=1.0 / 255.0,
            channel_first=True,
            add_batch_dimension=True,
        )
    )


def _build_tracker_config(
    activation_threshold: Any,
    lost_track_buffer: Any,
    matching_threshold: Any,
    frame_rate: Any,
    min_consecutive_frames: Any,
) -> TrackerConfig:
    def _val(v: Any, default: Any, fn: Any) -> Any:
        if hasattr(v, "default"):
            v = v.default
        try:
            return fn(v)
        except Exception:
            return default

    try:
        return TrackerConfig(
            track_activation_threshold=_val(activation_threshold, 0.25, float),
            lost_track_buffer=_val(lost_track_buffer, 30, int),
            minimum_matching_threshold=_val(matching_threshold, 0.8, float),
            frame_rate=_val(frame_rate, 30, int),
            minimum_consecutive_frames=_val(min_consecutive_frames, 1, int),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tracker config: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /api/tracking/video — stateless video upload tracking
# ---------------------------------------------------------------------------

@router.post("/video")
async def video_tracking(
    model_name: str = Form(...),
    file: UploadFile = File(...),
    max_frames: int = Form(300),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    # ByteTracker config params (exposed as form fields with sensible defaults)
    tracker_activation_threshold: float = Form(0.25),
    tracker_lost_track_buffer: int = Form(30),
    tracker_matching_threshold: float = Form(0.8),
    tracker_frame_rate: int = Form(30),
    tracker_min_consecutive_frames: int = Form(1),
    batch_size: int = Form(settings.batch_size),
) -> dict:
    """
    Run ByteTrack multi-object tracking on an uploaded video using ONNX batching (2 to 6 frames at once).

    A **fresh** tracker is created per request (stateless).
    Returns per-frame tracked objects with persistent ``track_id`` values.
    """

    try:
        conf_threshold = float(getattr(conf_threshold, "default", conf_threshold))
    except Exception:
        conf_threshold = 0.25

    try:
        iou_threshold = float(getattr(iou_threshold, "default", iou_threshold))
    except Exception:
        iou_threshold = 0.45

    try:
        max_frames = int(getattr(max_frames, "default", max_frames))
    except Exception:
        max_frames = 300

    try:
        batch_size = int(getattr(batch_size, "default", batch_size))
    except Exception:
        batch_size = settings.batch_size

    if not isinstance(batch_size, int) or batch_size < 2 or batch_size > 6:
        raise HTTPException(status_code=400, detail="batch_size must be an integer between 2 and 6.")

    # ── Model validation ────────────────────────────────────────────────
    model_name = (model_name or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required.")

    # ── File validation ──────────────────────────────────────────────────
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="Video file is required.")

    extension = Path(file.filename).suffix.lower()
    if not extension:
        raise HTTPException(status_code=415, detail="Uploaded video must have a file extension.")
    if extension not in settings.allowed_video_extensions:
        raise HTTPException(status_code=415, detail=f"Unsupported video type: {extension}")

    # ── Frame limit ──────────────────────────────────────────────────────
    if max_frames < 1 or max_frames > 3000:
        raise HTTPException(status_code=400, detail="max_frames must be between 1 and 3000.")

    # ── Read upload ──────────────────────────────────────────────────────
    try:
        data = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to read uploaded video.") from exc

    if not data:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded video exceeds the maximum allowed size of {settings.max_upload_size_mb} MB.",
        )

    # ── Tracker config ───────────────────────────────────────────────────
    tracker_config = _build_tracker_config(
        activation_threshold=tracker_activation_threshold,
        lost_track_buffer=tracker_lost_track_buffer,
        matching_threshold=tracker_matching_threshold,
        frame_rate=tracker_frame_rate,
        min_consecutive_frames=tracker_min_consecutive_frames,
    )

    temp_path: Path | None = None

    try:
        # ── Write to temp file ───────────────────────────────────────────
        settings.temp_directory.mkdir(parents=True, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(
                dir=settings.temp_directory,
                suffix=extension,
                prefix="track_",
                delete=False,
            ) as tmp:
                tmp.write(data)
                tmp.flush()
                temp_path = Path(tmp.name)
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Unable to create temporary video file.") from exc

        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Temporary video file could not be created.")

        # ── Per-request tracker & server-side annotation ─────────────────
        tracker = ByteTrackerWrapper(tracker_config)

        session_video_id = uuid.uuid4().hex[:12]
        annotated_dir = settings.temp_directory / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        raw_annotated_path = settings.temp_directory / f"raw_ann_{session_video_id}.mp4"
        final_annotated_path = annotated_dir / f"annotated_{session_video_id}.mp4"
        frames_dir = settings.temp_directory / "annotated_frames" / session_video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        writer: cv2.VideoWriter | None = None
        has_written_frames = False

        try:
            tw, th = _get_model_input_size(model_name)
            preprocessor = _create_preprocessor(target_width=tw, target_height=th)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Could not initialise preprocessor.") from exc

        frame_results: list[dict] = []
        frames_processed = 0
        session_alerted_suspects: set[str] = set()
        suspects_detected_summary: list[dict] = []
        face_scan_interval = 15

        # ── Open video ───────────────────────────────────────────────────
        with VideoLoader(temp_path) as video:
            try:
                metadata = video.get_metadata()
            except VideoLoaderError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            eof = False
            while frames_processed < max_frames and not eof:
                batch_frames: list[np.ndarray] = []
                batch_indices: list[int] = []
                batch_sizes: list[tuple[int, int]] = []
                target_count = min(batch_size, max_frames - frames_processed)

                while len(batch_frames) < target_count:
                    try:
                        success, frame = video.read_frame()
                    except VideoLoaderError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                    except Exception as exc:
                        raise HTTPException(
                            status_code=500,
                            detail="Unexpected error while reading video frame.",
                        ) from exc

                    if not success or frame is None:
                        eof = True
                        break

                    if not isinstance(frame, np.ndarray) or frame.size == 0:
                        raise HTTPException(status_code=500, detail="VideoLoader returned an invalid frame.")

                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise HTTPException(status_code=400, detail="Video frame must be a 3-channel color image.")

                    orig_h, orig_w = frame.shape[:2]
                    batch_indices.append(frames_processed + len(batch_frames))
                    batch_frames.append(frame)
                    batch_sizes.append((orig_w, orig_h))

                if not batch_frames:
                    break

                # ── Preprocess batch ─────────────────────────────────────
                try:
                    if len(batch_frames) == 1:
                        tensor = preprocessor.process(batch_frames[0])
                    else:
                        tensor = preprocessor.process_batch(batch_frames)
                except PreprocessingError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                # ── Batch Detect (2 to 6 frames at once) ─────────────────
                try:
                    inference_result = inference_service.predict(
                        model_name=model_name,
                        input_data=tensor,
                        postprocess=True,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        original_image_size=batch_sizes[0] if len(batch_sizes) == 1 else None,
                        original_image_sizes=batch_sizes if len(batch_sizes) > 1 else None,
                    )
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc)) from exc
                except Exception as exc:
                    raise HTTPException(status_code=500, detail="Video frame inference failed.") from exc

                if len(batch_frames) > 1:
                    batch_dets = inference_result.get("batch_detections")
                    if not batch_dets:
                        batch_dets = [inference_result.get("detections", []) for _ in batch_frames]
                else:
                    batch_dets = [inference_result.get("detections", [])]

                # ── Sequential Tracking & Annotation ─────────────────────
                for frame, f_idx, (orig_w, orig_h), raw_detections in zip(batch_frames, batch_indices, batch_sizes, batch_dets):
                    try:
                        tracked_objects = tracker.update(
                            detections=raw_detections,
                            frame_resolution=(orig_w, orig_h),
                        )
                    except ByteTrackerError as exc:
                        raise HTTPException(status_code=500, detail=str(exc)) from exc

                    tracked_dicts = [t.to_dict() for t in tracked_objects]

                    # ── Global Person Re-ID Assignment across Cameras ────────
                    for td in tracked_dicts:
                        c_name = str(td.get("class_name", "")).lower()
                        if (c_name in ("person", "human", "pedestrian") or td.get("class_id") == 0) and td.get("track_id") is not None:
                            try:
                                gtid, is_cross, r_score = global_trace_manager.update_track(
                                    camera_id=f"video_upload_{session_video_id}",
                                    camera_name=f"Video Analysis ({file.filename or 'Upload'})",
                                    local_track_id=td["track_id"],
                                    box=td.get("box", []),
                                    frame_bgr=frame,
                                    confidence=float(td.get("confidence", 0.8)),
                                )
                                td["global_trace_id"] = gtid
                                td["is_cross_camera"] = is_cross
                                td["reid_score"] = r_score
                            except Exception as reid_err:
                                logger.debug("ReID update skip in video tracking: %s", reid_err)

                    # ── Periodic Biometric Suspect Facial Scan ───────────────
                    if f_idx % face_scan_interval == 0:
                        try:
                            person_boxes = [
                                td.get("box") for td in tracked_dicts
                                if (str(td.get("class_name", "")).lower() in ("person", "human", "pedestrian") or td.get("class_id") == 0)
                                and td.get("box")
                            ]
                            detected_faces = face_service.detect_and_recognize(
                                frame,
                                min_match_score=None,
                                min_face_size=12,
                                det_score_thresh=0.25,
                                use_temporal_smoothing=True,
                                enable_distant_mode=True,
                                person_boxes=person_boxes if person_boxes else None,
                            )
                            for f in detected_faces:
                                is_threat = bool(f.get("is_threat", False))
                                suspect_name = f.get("name")
                                is_known = bool(f.get("is_known", False))

                                if is_threat or is_known:
                                    conf = float(f.get("calibrated_conf") or f.get("match_score") or f.get("confidence") or 0.85)
                                    face_dict = {
                                        "box": [float(c) for c in f["bbox"]],
                                        "confidence": round(conf, 3),
                                        "class_id": 999,
                                        "class_name": f"suspect_{suspect_name.lower()}" if is_threat else f"face_{suspect_name.lower()}",
                                        "is_threat": is_threat,
                                        "threat_level": f.get("threat_level") or ("HIGH" if is_threat else None),
                                        "suspect_name": suspect_name,
                                        "category": f.get("category"),
                                    }
                                    tracked_dicts.append(face_dict)

                                    if is_threat and suspect_name and suspect_name not in session_alerted_suspects:
                                        session_alerted_suspects.add(suspect_name)
                                        snap_name = f"suspect_vid_{session_video_id}_{f_idx}.jpg"
                                        snap_path = SNAPSHOTS_DIR / snap_name
                                        try:
                                            cv2.imwrite(str(snap_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                                        except Exception:
                                            snap_name = None

                                        t_level = f.get("threat_level") or "HIGH"
                                        sev = "Critical" if t_level == "CRITICAL" else "High"
                                        alert_service.create_alert(
                                            title=f"SUSPECT DETECTED: {suspect_name.upper()}",
                                            location=f"Video Analysis Stream ({file.filename or 'Upload'})",
                                            severity=sev,
                                            camera_id=f"video_upload_{session_video_id}",
                                            camera_name=f"Video Upload ({file.filename or 'Video'})",
                                            class_name="suspect",
                                            confidence=conf,
                                            box=[float(c) for c in f["bbox"]],
                                            snapshot_filename=snap_name,
                                            suspect_name=suspect_name,
                                            threat_level=t_level,
                                            category=f.get("category"),
                                            notes=f"Identified in uploaded video at frame {f_idx} with {int(conf * 100)}% biometric match confidence.",
                                        )
                                        suspects_detected_summary.append({
                                            "name": suspect_name,
                                            "threat_level": t_level,
                                            "frame_index": f_idx,
                                            "confidence": round(conf, 3),
                                            "category": f.get("category"),
                                        })
                        except Exception as face_err:
                            logger.debug("Face scan error during video tracking: %s", face_err)

                    # ── Server-Side Bounding Box & Track ID Creation ─────────
                    annotated_frame = frame.copy()
                    draw_tracked_boxes(
                        annotated_frame,
                        tracked_dicts,
                        frame_idx=f_idx,
                        draw_hud=True,
                    )

                    if writer is None:
                        fps_val = float(metadata.get("fps") or 25.0)
                        if fps_val <= 0 or fps_val > 120:
                            fps_val = 25.0
                        writer = cv2.VideoWriter(
                            str(raw_annotated_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps_val,
                            (orig_w, orig_h),
                        )

                    if writer is not None and writer.isOpened():
                        writer.write(annotated_frame)
                        has_written_frames = True

                    # Save individual frame JPEG for frame inspector
                    frame_jpg_path = frames_dir / f"frame_{f_idx:05d}.jpg"
                    try:
                        cv2.imwrite(str(frame_jpg_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    except Exception:
                        pass

                    frame_results.append(
                        {
                            "frame_index": f_idx,
                            "tracked_objects": tracked_dicts,
                            "detections_count": len(raw_detections),
                            "tracks_count": len(tracked_objects),
                            "annotated_frame_url": f"/api/tracking/video/frame/{session_video_id}/{f_idx}",
                        }
                    )

                    frames_processed += 1

        if writer is not None:
            writer.release()
            writer = None

        annotated_video_url: str | None = None
        if has_written_frames and raw_annotated_path.exists():
            success = convert_video_to_h264(raw_annotated_path, final_annotated_path)
            if success and final_annotated_path.exists():
                annotated_video_url = f"/api/tracking/video/annotated/annotated_{session_video_id}.mp4"
            if raw_annotated_path.exists():
                try:
                    raw_annotated_path.unlink()
                except OSError:
                    pass

        return {
            "model_name": model_name,
            "status": "success",
            "video": metadata,
            "video_id": session_video_id,
            "frames_requested": max_frames,
            "frames_processed": frames_processed,
            "batch_size": batch_size,
            "annotated_video_url": annotated_video_url,
            "suspects_detected": suspects_detected_summary,
            "tracker_config": {
                "track_activation_threshold": tracker_config.track_activation_threshold,
                "lost_track_buffer": tracker_config.lost_track_buffer,
                "minimum_matching_threshold": tracker_config.minimum_matching_threshold,
                "frame_rate": tracker_config.frame_rate,
                "minimum_consecutive_frames": tracker_config.minimum_consecutive_frames,
            },
            "results": frame_results,
        }

    except HTTPException:
        raise

    except VideoLoaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Unexpected error in video tracking.")
        raise HTTPException(status_code=500, detail="Video tracking failed.") from exc

    finally:
        if temp_path is not None:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# POST /api/tracking/rtsp — stateful RTSP tracking
# ---------------------------------------------------------------------------

@router.post("/rtsp")
async def rtsp_tracking(
    model_name: str = Form(...),
    rtsp_url: str = Form(...),
    camera_id: str = Form(...),
    max_frames: int = Form(5),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    # ByteTracker config (only applied when creating a NEW session for camera_id)
    tracker_activation_threshold: float = Form(0.25),
    tracker_lost_track_buffer: int = Form(30),
    tracker_matching_threshold: float = Form(0.8),
    tracker_frame_rate: int = Form(30),
    tracker_min_consecutive_frames: int = Form(1),
) -> dict:
    """
    Run ByteTrack tracking on a live RTSP stream.

    The tracker is **stateful**: track IDs persist across multiple requests
    for the same ``camera_id``.  Use ``DELETE /api/tracking/rtsp/{camera_id}``
    to explicitly reset track state.

    Processes a bounded number of frames per call (``max_frames``) to prevent
    unbounded HTTP requests.
    """

    # ── Validation ───────────────────────────────────────────────────────
    model_name = (model_name or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required.")

    try:
        conf_threshold = float(getattr(conf_threshold, "default", conf_threshold))
    except Exception:
        conf_threshold = 0.25

    try:
        iou_threshold = float(getattr(iou_threshold, "default", iou_threshold))
    except Exception:
        iou_threshold = 0.45

    try:
        max_frames = int(getattr(max_frames, "default", max_frames))
    except Exception:
        max_frames = 5

    rtsp_url = (rtsp_url or "").strip()
    if not rtsp_url:
        raise HTTPException(status_code=400, detail="rtsp_url is required.")
    if not rtsp_url.lower().startswith("rtsp://"):
        raise HTTPException(status_code=400, detail="rtsp_url must start with rtsp://")

    camera_id = (camera_id or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id is required.")

    if max_frames < 1 or max_frames > 100:
        raise HTTPException(status_code=400, detail="max_frames must be between 1 and 100.")

    # ── Tracker config (only used if this is a new session) ──────────────
    tracker_config = _build_tracker_config(
        activation_threshold=tracker_activation_threshold,
        lost_track_buffer=tracker_lost_track_buffer,
        matching_threshold=tracker_matching_threshold,
        frame_rate=tracker_frame_rate,
        min_consecutive_frames=tracker_min_consecutive_frames,
    )

    # ── Retrieve / create per-camera tracker ─────────────────────────────
    try:
        tracker = tracker_session_store.get_or_create(camera_id, config=tracker_config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stream: RTSPStream | None = None

    try:
        # ── RTSP connection ───────────────────────────────────────────────
        try:
            stream = RTSPStream(
                rtsp_url=rtsp_url,
                reconnect_attempts=3,
                reconnect_delay_seconds=1.0,
                connection_timeout_ms=5000,
                read_timeout_ms=5000,
            )
        except (InvalidRTSPUrlError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            stream.connect()
        except RTSPConnectionError as exc:
            raise HTTPException(status_code=503, detail=f"Unable to connect to RTSP stream: {exc}") from exc

        # ── Preprocessor ─────────────────────────────────────────────────
        try:
            tw, th = _get_model_input_size(model_name)
            preprocessor = _create_preprocessor(target_width=tw, target_height=th)
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Unable to initialise RTSP preprocessor.") from exc

        frame_results: list[dict] = []
        frames_processed = 0
        reconnect_count = 0

        # ── Bounded frame loop ────────────────────────────────────────────
        while frames_processed < max_frames:
            try:
                frame = stream.read_frame()
            except RTSPReadError as exc:
                reconnect_count += 1
                if reconnect_count > 1:
                    raise HTTPException(
                        status_code=503,
                        detail="RTSP frame read failed after reconnection attempt.",
                    ) from exc
                try:
                    reconnected = stream.reconnect()
                except RTSPError as reconnect_exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"RTSP stream disconnected and reconnection failed: {reconnect_exc}",
                    ) from reconnect_exc
                if not reconnected:
                    raise HTTPException(status_code=503, detail="RTSP stream disconnected and reconnection failed.")
                continue

            if frame is None:
                break

            if not isinstance(frame, np.ndarray) or frame.size == 0:
                raise HTTPException(status_code=500, detail="RTSP stream returned an invalid frame.")

            if frame.ndim != 3 or frame.shape[2] != 3:
                raise HTTPException(status_code=400, detail="RTSP frame must be a 3-channel color image.")

            # ── Preprocess ───────────────────────────────────────────────
            try:
                tensor = preprocessor.process(frame)
            except PreprocessingError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            # ── Detect ───────────────────────────────────────────────────
            orig_h, orig_w = frame.shape[:2]
            try:
                inference_result = inference_service.predict(
                    model_name=model_name,
                    input_data=tensor,
                    postprocess=True,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    original_image_size=(orig_w, orig_h),
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail="RTSP frame inference failed.") from exc

            raw_detections: list[dict] = inference_result.get("detections", [])

            # ── Track ─────────────────────────────────────────────────────
            try:
                tracked_objects = tracker.update(
                    detections=raw_detections,
                    frame_resolution=(orig_w, orig_h),
                )
            except ByteTrackerError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            # Record touch so session store stats are accurate
            tracker_session_store.touch(camera_id)

            tracked_dicts = [t.to_dict() for t in tracked_objects]
            for td in tracked_dicts:
                c_name = str(td.get("class_name", "")).lower()
                if (c_name in ("person", "human", "pedestrian") or td.get("class_id") == 0) and td.get("track_id") is not None:
                    try:
                        gtid, is_cross, r_score = global_trace_manager.update_track(
                            camera_id=camera_id,
                            camera_name=f"Camera {camera_id}",
                            local_track_id=td["track_id"],
                            box=td.get("box", []),
                            frame_bgr=frame,
                            confidence=float(td.get("confidence", 0.8)),
                        )
                        td["global_trace_id"] = gtid
                        td["is_cross_camera"] = is_cross
                        td["reid_score"] = r_score
                    except Exception as reid_err:
                        logger.debug("ReID update skip in RTSP tracking: %s", reid_err)

            # ── Geofence & Directional Tripwire Evaluation ────────────────
            try:
                from app.pipeline.geofence_engine import geofence_engine
                geofence_engine.evaluate_tracks(
                    camera_id=camera_id,
                    tracked_objects=tracked_dicts,
                    frame_resolution=(orig_w, orig_h),
                    auto_alert=True,
                    frame=frame,
                    camera_name=f"Camera {camera_id}",
                )
            except Exception as geo_err:
                logger.debug("Geofence evaluation skip in RTSP tracking: %s", geo_err)

            frame_results.append(
                {
                    "frame_index": frames_processed,
                    "tracked_objects": tracked_dicts,
                    "detections_count": len(raw_detections),
                    "tracks_count": len(tracked_objects),
                }
            )

            frames_processed += 1

        return {
            "model_name": model_name,
            "status": "success",
            "camera_id": camera_id,
            "rtsp_url": rtsp_url,
            "frames_requested": max_frames,
            "frames_processed": frames_processed,
            "tracker_frame_index": tracker.frame_index,
            "results": frame_results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Unexpected error in RTSP tracking | camera_id=%s", camera_id)
        raise HTTPException(status_code=500, detail="RTSP tracking failed.") from exc

    finally:
        if stream is not None:
            try:
                stream.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# GET /api/tracking/rtsp/sessions — list active tracker sessions
# ---------------------------------------------------------------------------

@router.get("/rtsp/sessions")
async def list_rtsp_sessions() -> dict:
    """List all active per-camera ByteTracker sessions."""
    sessions = tracker_session_store.list_sessions()
    return {
        "status": "success",
        "session_count": len(sessions),
        "sessions": sessions,
    }


# ---------------------------------------------------------------------------
# DELETE /api/tracking/rtsp/{camera_id} — reset a specific tracker session
# ---------------------------------------------------------------------------

@router.delete("/rtsp/{camera_id}")
async def reset_rtsp_session(camera_id: str) -> dict:
    """
    Reset and remove the ByteTracker session for the given ``camera_id``.

    After this call, the next request with the same ``camera_id`` will start
    with a fresh tracker and track IDs will restart from 1.
    """
    camera_id = (camera_id or "").strip()
    if not camera_id:
        raise HTTPException(status_code=400, detail="camera_id cannot be empty.")

    removed = tracker_session_store.reset(camera_id)

    if not removed:
        raise HTTPException(
            status_code=404,
            detail=f"No active tracker session found for camera_id='{camera_id}'.",
        )

    return {
        "status": "success",
        "message": f"Tracker session for camera_id='{camera_id}' has been reset.",
    }


# ---------------------------------------------------------------------------
# Server-side Annotated Video & Frame File Serving
# ---------------------------------------------------------------------------

@router.api_route("/video/annotated/{filename}", methods=["GET", "HEAD"])
async def get_annotated_video(filename: str):
    """
    Serve a server-annotated ByteTrack video with burned-in bounding boxes (H.264 MP4).
    Supports GET, HEAD, HTTP 206 Partial Content range requests, and inline playback.
    """
    safe_name = Path(filename).name
    target_path = settings.temp_directory / "annotated" / safe_name
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Annotated video not found.")
    return FileResponse(
        str(target_path),
        media_type="video/mp4",
        content_disposition_type="inline",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.api_route("/video/frame/{video_id}/{frame_index}", methods=["GET", "HEAD"])
async def get_annotated_frame(video_id: str, frame_index: int):
    """
    Serve a single server-annotated JPEG frame from tracked video.
    """
    safe_id = Path(video_id).name
    frame_path = settings.temp_directory / "annotated_frames" / safe_id / f"frame_{frame_index:05d}.jpg"
    if not frame_path.exists() or not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Annotated frame not found.")
    return FileResponse(
        str(frame_path),
        media_type="image/jpeg",
        content_disposition_type="inline",
        headers={
            "Cache-Control": "public, max-age=3600",
        },
    )


# ---------------------------------------------------------------------------
# Global Multi-Camera Person Re-ID Endpoints
# ---------------------------------------------------------------------------

@router.get("/global/traces")
async def list_global_traces():
    """List all active cross-camera person traces with recent sightings and trajectory counts."""
    traces = global_trace_manager.list_active_traces()
    return {
        "status": "success",
        "total_active": len(traces),
        "traces": traces,
    }


@router.get("/global/traces/{trace_id}")
async def get_global_trace_trajectory(trace_id: str):
    """Retrieve full chronological GPS multi-camera trajectory for a specific Global Trace ID."""
    traj = global_trace_manager.get_trace_trajectory(trace_id)
    if not traj.get("found"):
        raise HTTPException(status_code=404, detail=f"Global trace '{trace_id}' not found or expired.")
    return traj


@router.delete("/global/traces")
async def reset_global_traces():
    """Clear and reset all global cross-camera trace records."""
    global_trace_manager.reset()
    return {"status": "success", "message": "All global traces have been reset."}

