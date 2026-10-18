import uuid
from pathlib import Path
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.runtime import (
    inference_service,
    model_manager,
    resource_manager,
)
from app.pipeline.preprocessor import (
    PreprocessingConfig,
    PreprocessingError,
)
from app.pipeline.video_loader import (
    VideoLoader,
    VideoLoaderError,
)
from app.pipeline.rtsp_loader import (
    InvalidRTSPUrlError,
    RTSPConnectionError,
    RTSPError,
    RTSPReadError,
    RTSPStream,
)
from app.pipeline.face_service import face_service
from app.pipeline.alert_service import alert_service, SNAPSHOTS_DIR


router = APIRouter(
    prefix="/api/inference",
    tags=["Inference"],
)


def get_model_input_size(model_name: str) -> tuple[int, int]:
    """
    Get the preferred (width, height) resolution for a model.
    Defaults to (640, 640) for YOLO models or uses model metadata.
    """
    try:
        engine = model_manager.get_model(model_name)
        if getattr(engine, "input_size", None):
            return engine.input_size
    except Exception:
        pass
    return (640, 640)


def create_preprocessor(target_width: int = 640, target_height: int = 640):
    """
    Create the preprocessing pipeline with model-specific input resolution.
    """

    from app.pipeline.preprocessor import Preprocessor

    config = PreprocessingConfig(
        target_width=target_width,
        target_height=target_height,
        convert_bgr_to_rgb=True,
        normalize=False,
        scale=1.0 / 255.0,
        channel_first=True,
        add_batch_dimension=True,
    )

    return Preprocessor(config)


@router.get("/status")
async def inference_status() -> dict:
    """Return inference service and resource status."""

    try:
        return {
            "status": "ready",
            "resource_manager": resource_manager.get_status(),
            "models": model_manager.get_status(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve inference service status.",
        ) from exc


@router.post("/image")
async def image_inference(
    model_name: str = Form(...),
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    postprocess: bool = Form(True),
) -> dict:
    """Run inference on a single uploaded image."""

    # -------------------------
    # Model validation
    # -------------------------

    if not isinstance(model_name, str):
        raise HTTPException(
            status_code=400,
            detail="model_name must be a string.",
        )

    model_name = model_name.strip()

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="model_name is required.",
        )

    # -------------------------
    # File validation
    # -------------------------

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Image file is required.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    extension = Path(file.filename).suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=415,
            detail="Uploaded image must have a file extension.",
        )

    if extension not in settings.allowed_image_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type: {extension}",
        )

    # -------------------------
    # Read upload
    # -------------------------

    try:
        data = await file.read()

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read uploaded image.",
        ) from exc

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    # -------------------------
    # Size validation
    # -------------------------

    max_size = settings.max_upload_size_mb * 1024 * 1024

    if len(data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded image exceeds the maximum allowed size "
                f"of {settings.max_upload_size_mb} MB."
            ),
        )

    # -------------------------
    # Decode image
    # -------------------------

    try:
        image_array = np.frombuffer(
            data,
            dtype=np.uint8,
        )

        if image_array.size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image contains no data.",
            )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode uploaded image.",
        ) from exc

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        )

    if not isinstance(frame, np.ndarray):
        raise HTTPException(
            status_code=400,
            detail="Decoded image is not a valid NumPy array.",
        )

    if frame.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Decoded image is empty.",
        )

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image must be a 3-channel color image.",
        )

    # -------------------------
    # Preprocessing + inference
    # -------------------------

    try:
        tw, th = get_model_input_size(model_name)
        preprocessor = create_preprocessor(target_width=tw, target_height=th)

        tensor = preprocessor.process(frame)

        orig_h, orig_w = frame.shape[:2]

        return inference_service.predict(
            model_name=model_name,
            input_data=tensor,
            postprocess=postprocess,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            original_image_size=(orig_w, orig_h),
        )

    except PreprocessingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Image inference failed.",
        ) from exc


@router.post("/video")
async def video_inference(
    model_name: str = Form(...),
    file: UploadFile = File(...),
    max_frames: int = Form(300),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    postprocess: bool = Form(True),
    batch_size: int = Form(settings.batch_size),
) -> dict:
    """
    Run bounded inference on frames from an uploaded video using ONNX batching (2 to 6 frames at once).
    """

    # -------------------------
    # Model validation
    # -------------------------

    if not isinstance(model_name, str):
        raise HTTPException(
            status_code=400,
            detail="model_name must be a string.",
        )

    model_name = model_name.strip()

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="model_name is required.",
        )

    # -------------------------
    # File validation
    # -------------------------

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Video file is required.",
        )

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    # -------------------------
    # Frame limit validation
    # -------------------------

    if not isinstance(max_frames, int):
        raise HTTPException(
            status_code=400,
            detail="max_frames must be an integer.",
        )

    if max_frames < 1 or max_frames > 3000:
        raise HTTPException(
            status_code=400,
            detail="max_frames must be between 1 and 3000.",
        )

    # -------------------------
    # Batch size validation (2 to 6 frames at once)
    # -------------------------

    try:
        batch_size = int(getattr(batch_size, "default", batch_size))
    except Exception:
        batch_size = settings.batch_size

    if not isinstance(batch_size, int) or batch_size < 2 or batch_size > 6:
        raise HTTPException(
            status_code=400,
            detail="batch_size must be an integer between 2 and 6.",
        )

    # -------------------------
    # Extension validation
    # -------------------------

    extension = Path(file.filename).suffix.lower()

    if not extension:
        raise HTTPException(
            status_code=415,
            detail="Uploaded video must have a file extension.",
        )

    if extension not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type: {extension}",
        )

    # -------------------------
    # Read upload
    # -------------------------

    try:
        data = await file.read()

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read uploaded video.",
        ) from exc

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty.",
        )

    # -------------------------
    # Size validation
    # -------------------------

    max_size = settings.max_upload_size_mb * 1024 * 1024

    if len(data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded video exceeds the maximum allowed size "
                f"of {settings.max_upload_size_mb} MB."
            ),
        )

    temp_path = None
    metadata = None

    try:
        # -------------------------
        # Temporary directory
        # -------------------------

        try:
            settings.temp_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to create temporary directory.",
            ) from exc

        # -------------------------
        # Unique temporary file
        # -------------------------

        try:
            with tempfile.NamedTemporaryFile(
                dir=settings.temp_directory,
                suffix=extension,
                prefix="upload_",
                delete=False,
            ) as temp_file:

                temp_file.write(data)
                temp_file.flush()

                temp_path = Path(temp_file.name)

        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to create temporary video file.",
            ) from exc

        # -------------------------
        # Temporary file validation
        # -------------------------

        if not temp_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Temporary video file could not be created.",
            )

        if not temp_path.is_file():
            raise HTTPException(
                status_code=500,
                detail="Temporary video path is not a file.",
            )

        if temp_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Temporary video file is empty.",
            )

        # -------------------------
        # Open video
        # -------------------------

        with VideoLoader(temp_path) as video:

            try:
                metadata = video.get_metadata()

            except VideoLoaderError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            tw, th = get_model_input_size(model_name)
            preprocessor = create_preprocessor(target_width=tw, target_height=th)

            frame_results = []
            frames_processed = 0
            session_alerted_suspects: set[str] = set()
            suspects_detected_summary: list[dict] = []
            face_scan_interval = 15

            # -------------------------
            # Batched frame processing (2 to 6 frames at once)
            # -------------------------

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
                        raise HTTPException(
                            status_code=400,
                            detail=str(exc),
                        ) from exc

                    except Exception as exc:
                        raise HTTPException(
                            status_code=500,
                            detail="Unexpected error while reading video frame.",
                        ) from exc

                    if not success or frame is None:
                        eof = True
                        break

                    if not isinstance(frame, np.ndarray):
                        raise HTTPException(
                            status_code=500,
                            detail="VideoLoader returned an invalid frame type.",
                        )

                    if frame.size == 0:
                        raise HTTPException(
                            status_code=400,
                            detail="VideoLoader returned an empty frame.",
                        )

                    if frame.ndim != 3 or frame.shape[2] != 3:
                        raise HTTPException(
                            status_code=400,
                            detail=(
                                "VideoLoader returned a frame that is "
                                "not a 3-channel color image."
                            ),
                        )

                    orig_h, orig_w = frame.shape[:2]
                    batch_indices.append(frames_processed + len(batch_frames))
                    batch_frames.append(frame)
                    batch_sizes.append((orig_w, orig_h))

                if not batch_frames:
                    break

                # -------------------------
                # Preprocessing
                # -------------------------

                try:
                    if len(batch_frames) == 1:
                        tensor = preprocessor.process(batch_frames[0])
                    else:
                        tensor = preprocessor.process_batch(batch_frames)

                except PreprocessingError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=str(exc),
                    ) from exc

                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail="Video frame preprocessing failed.",
                    ) from exc

                # -------------------------
                # Inference
                # -------------------------

                try:
                    batch_inf_result = inference_service.predict(
                        model_name=model_name,
                        input_data=tensor,
                        postprocess=postprocess,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        original_image_size=batch_sizes[0] if len(batch_sizes) == 1 else None,
                        original_image_sizes=batch_sizes if len(batch_sizes) > 1 else None,
                    )

                except KeyError as exc:
                    raise HTTPException(
                        status_code=404,
                        detail=str(exc),
                    ) from exc

                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail="Video frame inference failed.",
                    ) from exc

                if len(batch_frames) > 1:
                    batch_dets = batch_inf_result.get("batch_detections")
                    if not batch_dets:
                        batch_dets = [batch_inf_result.get("detections", []) for _ in batch_frames]
                else:
                    batch_dets = [batch_inf_result.get("detections", [])]

                frame_inf_time = round(batch_inf_result.get("inference_time_ms", 0.0) / len(batch_frames), 2)

                for frame, f_idx, (orig_w, orig_h), detections in zip(batch_frames, batch_indices, batch_sizes, batch_dets):
                    frame_inf = {
                        "model_name": model_name,
                        "status": "success",
                        "inference_time_ms": frame_inf_time,
                        "detections_count": len(detections),
                        "detections": detections,
                    }

                    # ── Periodic Biometric Suspect Facial Scan ───────────────
                    if f_idx % face_scan_interval == 0:
                        try:
                            person_boxes = [
                                d.get("box") for d in detections
                                if (str(d.get("class_name", "")).lower() in ("person", "human", "pedestrian") or d.get("class_id") == 0)
                                and d.get("box")
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
                            raw_dets = list(detections)
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
                                    raw_dets.append(face_dict)

                                    if is_threat and suspect_name and suspect_name not in session_alerted_suspects:
                                        session_alerted_suspects.add(suspect_name)
                                        session_id_short = uuid.uuid4().hex[:8]
                                        snap_name = f"suspect_inf_{session_id_short}_{f_idx}.jpg"
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
                                            camera_id="video_upload_stream",
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
                            frame_inf["detections"] = raw_dets
                            frame_inf["detections_count"] = len(raw_dets)
                        except Exception:
                            pass

                    frame_results.append(
                        {
                            "frame_index": f_idx,
                            "inference": frame_inf,
                        }
                    )
                    frames_processed += 1

        # -------------------------
        # Video response
        # -------------------------

        return {
            "model_name": model_name,
            "status": "success",
            "video": metadata,
            "frames_requested": max_frames,
            "frames_processed": frames_processed,
            "batch_size": batch_size,
            "suspects_detected": suspects_detected_summary,
            "results": frame_results,
        }

    except HTTPException:
        raise

    except VideoLoaderError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Video inference failed.",
        ) from exc

    finally:
        # -------------------------
        # Always cleanup temp file
        # -------------------------

        if temp_path is not None:

            try:
                if temp_path.exists():
                    temp_path.unlink()

            except OSError:
                pass


@router.post("/rtsp")
async def rtsp_inference(
    model_name: str = Form(...),
    rtsp_url: str = Form(...),
    max_frames: int = Form(5),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    postprocess: bool = Form(True),
    batch_size: int = Form(settings.batch_size),
) -> dict:
    """
    Run bounded inference on an RTSP CCTV stream using ONNX batching (2 to 6 frames at once).

    The endpoint processes a limited number of frames per request
    to prevent an unbounded HTTP request from running forever.
    """

    # -------------------------
    # Model validation
    # -------------------------

    if not isinstance(model_name, str):
        raise HTTPException(
            status_code=400,
            detail="model_name must be a string.",
        )

    model_name = model_name.strip()

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="model_name is required.",
        )

    # -------------------------
    # RTSP URL validation
    # -------------------------

    if not isinstance(rtsp_url, str):
        raise HTTPException(
            status_code=400,
            detail="rtsp_url must be a string.",
        )

    rtsp_url = rtsp_url.strip()

    if not rtsp_url:
        raise HTTPException(
            status_code=400,
            detail="rtsp_url is required.",
        )

    if not rtsp_url.lower().startswith("rtsp://"):
        raise HTTPException(
            status_code=400,
            detail="rtsp_url must start with rtsp://",
        )

    # -------------------------
    # Frame limit validation
    # -------------------------

    if not isinstance(max_frames, int):
        raise HTTPException(
            status_code=400,
            detail="max_frames must be an integer.",
        )

    if max_frames < 1 or max_frames > 100:
        raise HTTPException(
            status_code=400,
            detail="max_frames must be between 1 and 100.",
        )

    # -------------------------
    # Batch size validation (2 to 6 frames at once)
    # -------------------------

    try:
        batch_size = int(getattr(batch_size, "default", batch_size))
    except Exception:
        batch_size = settings.batch_size

    if not isinstance(batch_size, int) or batch_size < 2 or batch_size > 6:
        raise HTTPException(
            status_code=400,
            detail="batch_size must be an integer between 2 and 6.",
        )

    stream = None

    try:
        # -------------------------
        # Create RTSP stream
        # -------------------------

        try:
            stream = RTSPStream(
                rtsp_url=rtsp_url,
                reconnect_attempts=3,
                reconnect_delay_seconds=1.0,
                connection_timeout_ms=5000,
                read_timeout_ms=5000,
            )

        except InvalidRTSPUrlError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        # -------------------------
        # Connect
        # -------------------------

        try:
            stream.connect()

        except RTSPConnectionError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Unable to connect to RTSP stream: {exc}",
            ) from exc

        # -------------------------
        # Preprocessor
        # -------------------------

        try:
            tw, th = get_model_input_size(model_name)
            preprocessor = create_preprocessor(target_width=tw, target_height=th)

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Unable to initialize RTSP preprocessor.",
            ) from exc

        frame_results = []
        frames_processed = 0
        reconnect_count = 0

        # -------------------------
        # Bounded RTSP loop with batch inference (2 to 6 frames at once)
        # -------------------------

        while frames_processed < max_frames:
            batch_frames: list[np.ndarray] = []
            batch_indices: list[int] = []
            batch_sizes: list[tuple[int, int]] = []
            target_count = min(batch_size, max_frames - frames_processed)

            while len(batch_frames) < target_count:
                try:
                    frame = stream.read_frame()

                except RTSPReadError as exc:
                    reconnect_count += 1

                    if reconnect_count > 1:
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                "RTSP frame read failed after "
                                "reconnection attempt."
                            ),
                        ) from exc

                    try:
                        reconnected = stream.reconnect()

                    except RTSPError as reconnect_exc:
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                "RTSP stream disconnected and "
                                f"reconnection failed: {reconnect_exc}"
                            ),
                        ) from reconnect_exc

                    if not reconnected:
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                "RTSP stream disconnected and "
                                "reconnection failed."
                            ),
                        )

                    continue

                except RTSPConnectionError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=str(exc),
                    ) from exc

                except RTSPError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=str(exc),
                    ) from exc

                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail="Unexpected RTSP frame-read error.",
                    ) from exc

                # -------------------------
                # Frame validation
                # -------------------------

                if frame is None:
                    raise HTTPException(
                        status_code=503,
                        detail="RTSP stream returned no frame.",
                    )

                if not isinstance(frame, np.ndarray):
                    raise HTTPException(
                        status_code=500,
                        detail="RTSP loader returned an invalid frame type.",
                    )

                if frame.size == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="RTSP stream returned an empty frame.",
                    )

                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "RTSP stream returned a frame that is "
                            "not a 3-channel color image."
                        ),
                    )

                orig_h, orig_w = frame.shape[:2]
                batch_indices.append(frames_processed + len(batch_frames))
                batch_frames.append(frame)
                batch_sizes.append((orig_w, orig_h))

            if not batch_frames:
                break

            # -------------------------
            # Preprocessing
            # -------------------------

            try:
                if len(batch_frames) == 1:
                    tensor = preprocessor.process(batch_frames[0])
                else:
                    tensor = preprocessor.process_batch(batch_frames)

            except PreprocessingError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=str(exc),
                ) from exc

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail="RTSP frame preprocessing failed.",
                ) from exc

            # -------------------------
            # Batch Inference
            # -------------------------

            try:
                batch_result = inference_service.predict(
                    model_name=model_name,
                    input_data=tensor,
                    postprocess=postprocess,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    original_image_size=batch_sizes[0] if len(batch_sizes) == 1 else None,
                    original_image_sizes=batch_sizes if len(batch_sizes) > 1 else None,
                )

            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail=str(exc),
                ) from exc

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail="RTSP frame inference failed.",
                ) from exc

            if len(batch_frames) > 1:
                batch_dets = batch_result.get("batch_detections")
                if not batch_dets:
                    batch_dets = [batch_result.get("detections", []) for _ in batch_frames]
            else:
                batch_dets = [batch_result.get("detections", [])]

            frame_inf_time = round(batch_result.get("inference_time_ms", 0.0) / len(batch_frames), 2)

            for f_idx, detections in zip(batch_indices, batch_dets):
                frame_results.append(
                    {
                        "frame_index": f_idx,
                        "inference": {
                            "model_name": model_name,
                            "status": "success",
                            "inference_time_ms": frame_inf_time,
                            "detections_count": len(detections),
                            "detections": detections,
                        },
                    }
                )
                frames_processed += 1

        # -------------------------
        # Success response
        # -------------------------

        return {
            "model_name": model_name,
            "status": "success",
            "batch_size": batch_size,
            "stream": {
                "rtsp_url_configured": True,
                "frames_requested": max_frames,
                "frames_processed": frames_processed,
                "reconnects": reconnect_count,
            },
            "results": frame_results,
        }

    except HTTPException:
        raise

    except RTSPError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="RTSP inference failed.",
        ) from exc

    finally:
        # -------------------------
        # Always release RTSP resource
        # -------------------------

        if stream is not None:

            try:
                stream.release()

            except Exception:
                pass


@router.post("/batch")
async def batch_inference(
    model_name: str = Form(...),
    files: list[UploadFile] = File(...),
    batch_size: int = Form(settings.batch_size),
    conf_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    postprocess: bool = Form(True),
) -> dict:
    """
    Run ONNX batch inference on multiple uploaded images in parallel chunks of 2 to 6 frames.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise HTTPException(status_code=400, detail="model_name is required.")

    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    try:
        batch_size = int(getattr(batch_size, "default", batch_size))
    except Exception:
        batch_size = settings.batch_size

    if not isinstance(batch_size, int) or batch_size < 2 or batch_size > 6:
        raise HTTPException(
            status_code=400,
            detail="batch_size must be an integer between 2 and 6.",
        )

    tw, th = get_model_input_size(model_name.strip())
    preprocessor = create_preprocessor(target_width=tw, target_height=th)

    # Decode uploaded images
    loaded_images: list[np.ndarray] = []
    filenames: list[str] = []
    for f in files:
        if not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in settings.allowed_image_extensions:
            raise HTTPException(status_code=415, detail=f"Unsupported image type: {ext} in {f.filename}")
        data = await f.read()
        if not data:
            continue
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3:
            continue
        loaded_images.append(frame)
        filenames.append(f.filename)

    if not loaded_images:
        raise HTTPException(status_code=400, detail="No valid images could be decoded.")

    results: list[dict] = []
    # Process in chunks of batch_size (2 to 6)
    for chunk_start in range(0, len(loaded_images), batch_size):
        chunk_frames = loaded_images[chunk_start : chunk_start + batch_size]
        chunk_names = filenames[chunk_start : chunk_start + batch_size]
        chunk_sizes = [(f.shape[1], f.shape[0]) for f in chunk_frames]

        try:
            if len(chunk_frames) == 1:
                tensor = preprocessor.process(chunk_frames[0])
            else:
                tensor = preprocessor.process_batch(chunk_frames)
        except PreprocessingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Batch image preprocessing failed.") from exc

        try:
            batch_inf_result = inference_service.predict(
                model_name=model_name.strip(),
                input_data=tensor,
                postprocess=postprocess,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                original_image_size=chunk_sizes[0] if len(chunk_sizes) == 1 else None,
                original_image_sizes=chunk_sizes if len(chunk_sizes) > 1 else None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail="Batch inference forward pass failed.") from exc

        if len(chunk_frames) > 1:
            batch_dets = batch_inf_result.get("batch_detections")
            if not batch_dets:
                batch_dets = [batch_inf_result.get("detections", []) for _ in chunk_frames]
        else:
            batch_dets = [batch_inf_result.get("detections", [])]

        chunk_time_each = round(batch_inf_result.get("inference_time_ms", 0.0) / len(chunk_frames), 2)
        for name, (orig_w, orig_h), dets in zip(chunk_names, chunk_sizes, batch_dets):
            results.append({
                "filename": name,
                "image_size": [orig_w, orig_h],
                "detections_count": len(dets),
                "inference_time_ms": chunk_time_each,
                "detections": dets,
            })

    return {
        "model_name": model_name.strip(),
        "status": "success",
        "batch_size": batch_size,
        "total_images": len(results),
        "results": results,
    }

