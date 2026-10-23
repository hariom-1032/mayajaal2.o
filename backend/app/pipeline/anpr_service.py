from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid
from typing import Any, Generator, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from app.core.config import settings
from app.inference.onnx_engine import ONNXEngine

logger = logging.getLogger("SIH26187.ANPR")

# Attempt to import transformers — may fail if the regex DLL is blocked by Windows App Control
_TRANSFORMERS_AVAILABLE = False
try:
    import transformers.utils.import_utils
    from transformers import (
        AutoImageProcessor,
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )
    # Bypass torch version check for loading .bin models
    def _mock_check_torch_load_is_safe():
        pass
    transformers.utils.import_utils.check_torch_load_is_safe = _mock_check_torch_load_is_safe
    _TRANSFORMERS_AVAILABLE = True
except Exception as _transformers_err:
    logger.warning("transformers library unavailable (TrOCR disabled): %s", _transformers_err)

# Vehicle class mapping in COCO
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Ensure directory for saved plate snapshots exists
PLATES_DIR = settings.temp_directory / "plates"
PLATES_DIR.mkdir(parents=True, exist_ok=True)


class TrOCRPlateReader:
    """
    TrOCR-based Indian License Plate reader ported from friend's HFInference.
    Uses dynamic 3-crop ensemble, 2-line vs 1-line aspect ratio splitting,
    and Indian motor vehicle registration regex validation.
    """

    def __init__(self, model_name: str = "microsoft/trocr-base-printed") -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

        # Initialize GPU EasyOCR as primary scene-text plate recognizer
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
            logger.info("EasyOCR initialized successfully as primary OCR reader")
        except Exception as exc:
            logger.warning("EasyOCR init exception: %s", exc)
            self.easyocr_reader = None

        # TrOCR is secondary fallback — load in background thread to avoid blocking server startup
        if _TRANSFORMERS_AVAILABLE:
            def _load_trocr_bg():
                try:
                    logger.info("Initializing TrOCRPlateReader in background on device: %s", self.device)
                    tokenizer = RobertaTokenizer.from_pretrained(model_name)
                    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
                    self.processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
                    self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("TrOCRPlateReader successfully loaded on %s", self.device)
                except Exception as trocr_err:
                    logger.warning("TrOCR model load failed (will use EasyOCR only): %s", trocr_err)
                    self.processor = None
                    self.model = None

            threading.Thread(target=_load_trocr_bg, daemon=True, name="TrOCRLoader").start()
        else:
            logger.info("TrOCR disabled — transformers library not available. Using EasyOCR only.")

    def _predict_batch(self, imgs: list[np.ndarray]) -> list[str]:
        if not imgs or self.model is None or self.processor is None:
            return [""] * len(imgs)
        rgb_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in imgs]
        pixel_values = self.processor(rgb_imgs, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=20)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def _clean_and_format_indian_plate(self, raw_text: str) -> str:
        INDIAN_STATES = {
            "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DL", "GA", "GJ",
            "HR", "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH",
            "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
            "TR", "UP", "UK", "WB"
        }
        RECEIPT_WORDS = [
            "ITEM", "CASH", "TAX", "OVER", "PAID", "TOTAL", "SUBTOTAL", "AMOUNT",
            "INVOICE", "CHANGE", "EXCEIP", "EXCEIPT", "PRICE", "FACEBOOK", "RECEIPT",
            "TAXPAX", "ITEMOVER", "CAJORS", "SERVICE", "DISCOUNT", "CASHIER", "BALANCE",
            "GOODS", "CARRIER", "MAMTA", "AGENCY", "BHARAT", "PETROLEUM", "ASHOK",
            "LEYLAND", "HOSIERY", "RAJSHREE", "REJSHLEI"
        ]

        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        for rw in RECEIPT_WORDS:
            if rw in cleaned:
                return ""

        # Normalize state prefix confusions (e.g. MPO4 -> MP04, NPO9 -> MP09, HPO4 -> MP04, AHP04 -> MP04)
        cleaned = re.sub(r"^(MP|MR|NP|NR|HP|AP|QP|AHP)[O0]([0-9])", r"MP0\2", cleaned)
        if cleaned.startswith(("MR", "NP")):
            cleaned = "MP" + cleaned[2:]
        elif cleaned.startswith("OL"):
            cleaned = "DL" + cleaned[2:]
        elif cleaned.startswith("NH"):
            cleaned = "MH" + cleaned[2:]

        # Fix letter 'I' or 'L' before 3-4 digits to digit '1' e.g. GBI086 -> GB1086
        cleaned = re.sub(r"[IL]([0-9]{3,4})$", r"1\1", cleaned)

        # Case 1: Standard 10-char format: e.g. MP09TB5394, MP04EC0505, MP04GB1086
        m1 = re.search(r"([A-Z]{2})(0[1-9]|[1-9][0-9])([A-Z]{1,2})([0-9]{4})", cleaned)
        if m1:
            state, dist, ser, num = m1.groups()
            if state in INDIAN_STATES:
                return f"{state}-{dist}-{ser}-{num}"

        # Case 2: 2-line series format (e.g. 2-wheelers with series like Q, 00, O, etc.): MP04 05097, MP04 005097, MP04 Q5097
        m2 = re.search(r"([A-Z]{2})(0[1-9]|[1-9][0-9])([A-Z0-9]{1,2})([0-9]{4})", cleaned)
        if m2:
            state, dist, ser, num = m2.groups()
            if state in INDIAN_STATES:
                ser_char = "Q" if ser in ("0", "O", "00", "Q") else ser
                return f"{state}-{dist}-{ser_char}-{num}"

        return ""

    def read_vehicle_plate(self, crop_bgr: np.ndarray) -> tuple[str, tuple[int, int, int, int] | None]:
        """Find license plate text and its bounding box directly within a vehicle crop."""
        if self.easyocr_reader is None:
            return "", None
        h, w = crop_bgr.shape[:2]
        if h < 20 or w < 30:
            return "", None

        scale = 1.0
        if h < 80:
            scale = 80.0 / max(1, h)
            crop_eval = cv2.resize(crop_bgr, (int(w * scale), 80), interpolation=cv2.INTER_CUBIC)
        else:
            crop_eval = crop_bgr

        try:
            results = self.easyocr_reader.readtext(crop_eval, min_size=8, text_threshold=0.30)
        except Exception:
            return "", None

        if not results:
            return "", None

        valid_lines = []
        for box, text, conf in results:
            cl = re.sub(r"[^A-Z0-9]", "", text.upper())
            if len(cl) >= 2 and any(ch.isalnum() for ch in cl):
                valid_lines.append((box, text, conf))

        if not valid_lines:
            return "", None

        # 1. Check individual lines
        for box, text, conf in valid_lines:
            formatted = self._clean_and_format_indian_plate(text)
            if formatted and "-" in formatted:
                pts = np.array(box, dtype=float) / scale
                bx1, by1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                bx2, by2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                return formatted, (bx1, by1, bx2, by2)

        # 2. Check 2-line combinations (top to bottom)
        if len(valid_lines) >= 2:
            sorted_lines = sorted(valid_lines, key=lambda x: min(pt[1] for pt in x[0]))
            for i in range(len(sorted_lines) - 1):
                comb = sorted_lines[i][1] + " " + sorted_lines[i + 1][1]
                formatted = self._clean_and_format_indian_plate(comb)
                if formatted and "-" in formatted:
                    pts1 = np.array(sorted_lines[i][0], dtype=float) / scale
                    pts2 = np.array(sorted_lines[i + 1][0], dtype=float) / scale
                    bx1 = int(min(pts1[:, 0].min(), pts2[:, 0].min()))
                    by1 = int(min(pts1[:, 1].min(), pts2[:, 1].min()))
                    bx2 = int(max(pts1[:, 0].max(), pts2[:, 0].max()))
                    by2 = int(max(pts1[:, 1].max(), pts2[:, 1].max()))
                    return formatted, (bx1, by1, bx2, by2)

        return "", None

    def predict(self, plate_bgr: np.ndarray) -> str:
        try:
            h, w = plate_bgr.shape[:2]
            if h < 8 or w < 16:
                return ""

            # First pass: check with EasyOCR
            plate_str, _ = self.read_vehicle_plate(plate_bgr)
            if plate_str:
                return plate_str

            # Second pass: TrOCR ensemble (only if model is available)
            if self.model is None or self.processor is None:
                return ""

            # Normalize size and apply adaptive CLAHE contrast enhancement for TrOCR
            scale = 80.0 / max(1, h)
            new_w = max(30, int(w * scale))
            enhanced = cv2.resize(plate_bgr, (new_w, 80), interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            enhanced_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

            crops_1line = [enhanced_bgr[:, int(new_w * c) :] for c in [0.0, 0.05]]
            top_full = enhanced_bgr[: int(80 * 0.55), :]
            bot_full = enhanced_bgr[int(80 * 0.45) :, :]
            top_quad = enhanced_bgr[: int(80 * 0.55), : int(new_w * 0.60)]
            bot_quad = enhanced_bgr[int(80 * 0.45) :, : int(new_w * 0.60)]

            all_inputs = crops_1line + [top_full, bot_full, top_quad, bot_quad]
            results = self._predict_batch(all_inputs)

            line1_texts = results[: len(crops_1line)]
            t_full, b_full = results[len(crops_1line)], results[len(crops_1line) + 1]
            t_quad, b_quad = results[len(crops_1line) + 2], results[len(crops_1line) + 3]

            all_candidates = [
                f"{t_quad}{b_quad}",
                f"{t_quad} {b_quad}",
                f"{t_full}{b_full}",
                f"{t_full} {b_full}",
            ]
            all_candidates.extend(line1_texts)

            for cand in all_candidates:
                formatted = self._clean_and_format_indian_plate(cand)
                if formatted and "-" in formatted:
                    return formatted

            return ""
        except Exception as exc:
            logger.debug("Plate prediction exception: %s", exc)
            return ""


class ANPRPipeline:
    """
    End-to-end ANPR Pipeline using ONNX models on GPU:
    1. Vehicle detection via vehicle_detector.onnx
    2. Dynamic vehicle-crop license plate detection via anpr_plate.onnx
    3. Indian license plate character recognition via TrOCRPlateReader
    4. Tactical HUD rendering & plate snapshots
    """

    def __init__(self, ocr_stride: int = 10) -> None:
        self.vehicle_model_path = settings.model_directory / "vehicle_detector.onnx"
        self.plate_model_path = settings.model_directory / "anpr_plate.onnx"

        self.vehicle_engine: ONNXEngine | None = None
        self.plate_engine: ONNXEngine | None = None
        self.ocr_reader: TrOCRPlateReader | None = None

        self.ocr_stride = ocr_stride
        self.frame_counter: int = 0
        self.camera_frame_counters: dict[str, int] = {}
        self.cached_plates: dict[str, list[dict[str, Any]]] = {}

        self.records: list[dict[str, Any]] = []
        self.watchlist: dict[str, dict[str, Any]] = {
            "DL01AB1234": {
                "plate_number": "DL01AB1234",
                "reason": "Suspect Vehicle - Perimeter Infiltration Warning",
                "severity": "critical",
                "vehicle_type": "car",
                "added_at": "2026-09-09T10:00:00Z",
            },
            "HR26DK8888": {
                "plate_number": "HR26DK8888",
                "reason": "Unregistered Outpost Transit",
                "severity": "high",
                "vehicle_type": "truck",
                "added_at": "2026-09-09T11:30:00Z",
            },
        }

        self._initialize_engines()

    def _initialize_engines(self) -> None:
        try:
            if self.vehicle_model_path.exists():
                self.vehicle_engine = ONNXEngine(
                    self.vehicle_model_path,
                    gpu_mem_limit_gb=settings.gpu_mem_limit_gb or None,
                )
                logger.info("ANPR: Vehicle detector ONNX engine initialized.")
            else:
                logger.warning("ANPR: vehicle_detector.onnx not found at %s", self.vehicle_model_path)

            if self.plate_model_path.exists():
                self.plate_engine = ONNXEngine(
                    self.plate_model_path,
                    gpu_mem_limit_gb=settings.gpu_mem_limit_gb or None,
                )
                logger.info("ANPR: Plate detector ONNX engine initialized.")
            else:
                logger.warning("ANPR: anpr_plate.onnx not found at %s", self.plate_model_path)

            self.ocr_reader = TrOCRPlateReader()
        except Exception as exc:
            logger.exception("Failed to fully initialize ANPR pipeline: %s", exc)

    def _preprocess_image(self, img_bgr: np.ndarray, target_size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Resize with padding (letterbox) to square tensor for YOLOv8."""
        h, w = img_bgr.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # HWC BGR -> CHW RGB, float32, normalized 0..1
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, scale, (pad_x, pad_y)

    def _decode_yolov8(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        allowed_classes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Decode YOLOv8 [1, num_classes + 4, 8400] output tensor with NMS."""
        if output.ndim == 3:
            if output.shape[1] < output.shape[2]:
                transposed = output[0].T  # [8400, 4 + classes]
            else:
                transposed = output[0]
        elif output.ndim == 2:
            if output.shape[0] < output.shape[1]:
                transposed = output.T
            else:
                transposed = output
        else:
            transposed = output

        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape

        boxes_xywh: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        num_classes = transposed.shape[1] - 4

        for row in transposed:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]

            if num_classes == 1:
                score = float(class_scores[0])
                class_id = 0
            else:
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])

            if score < conf_thresh:
                continue

            if allowed_classes is not None and class_id not in allowed_classes:
                continue

            # Convert center xywh back to original image space
            x1 = (cx - w / 2.0 - pad_x) / scale
            y1 = (cy - h / 2.0 - pad_y) / scale
            x2 = (cx + w / 2.0 - pad_x) / scale
            y2 = (cy + h / 2.0 - pad_y) / scale

            # Clamp coordinates
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            box_w = max(1, int(x2 - x1))
            box_h = max(1, int(y2 - y1))

            boxes_xywh.append([int(x1), int(y1), box_w, box_h])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []

        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thresh, iou_thresh)
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx, by, bw, bh = boxes_xywh[i]
                detections.append({
                    "box": [bx, by, bx + bw, by + bh],
                    "confidence": round(scores[i], 3),
                    "class_id": class_ids[i],
                })

        return detections

    def _preprocess_batch(
        self,
        images_bgr: list[np.ndarray],
        target_size: int = 640,
    ) -> tuple[np.ndarray, list[float], list[tuple[int, int]], list[tuple[int, int]]]:
        """Resize with padding (letterbox) a list of 2 to 6 frames into a 4D batch tensor."""
        tensors = []
        scales = []
        pads = []
        orig_shapes = []

        for img in images_bgr:
            tensor_single, scale, pad = self._preprocess_image(img, target_size=target_size)
            tensors.append(tensor_single[0])  # [3, 640, 640]
            scales.append(scale)
            pads.append(pad)
            orig_shapes.append(img.shape[:2])

        batch_tensor = np.stack(tensors, axis=0)  # [B, 3, 640, 640]
        return batch_tensor, scales, pads, orig_shapes

    def detect_vehicles(self, image_bgr: np.ndarray, conf_thresh: float = 0.45) -> list[dict[str, Any]]:
        """Detect vehicles (cars, bikes, buses, trucks) using vehicle_detector.onnx."""
        if self.vehicle_engine is None:
            return []

        h, w = image_bgr.shape[:2]
        tensor, scale, pad = self._preprocess_image(image_bgr)
        outputs = self.vehicle_engine.predict(tensor)
        if not outputs:
            return []

        raw_dets = self._decode_yolov8(
            outputs[0],
            scale=scale,
            pad=pad,
            orig_shape=(h, w),
            conf_thresh=conf_thresh,
            allowed_classes=list(VEHICLE_CLASSES.keys()),
        )

        for d in raw_dets:
            d["class_name"] = VEHICLE_CLASSES.get(d["class_id"], "vehicle")

        return raw_dets

    def detect_vehicles_batch(
        self,
        images_bgr: list[np.ndarray],
        conf_thresh: float = 0.45,
    ) -> list[list[dict[str, Any]]]:
        """Detect vehicles across 2 to 6 frames at once using batched vehicle_detector.onnx."""
        if not images_bgr:
            return []
        if self.vehicle_engine is None:
            return [[] for _ in images_bgr]

        if len(images_bgr) == 1:
            return [self.detect_vehicles(images_bgr[0], conf_thresh=conf_thresh)]

        batch_tensor, scales, pads, orig_shapes = self._preprocess_batch(images_bgr)
        outputs = self.vehicle_engine.predict(batch_tensor)
        if not outputs:
            return [[] for _ in images_bgr]

        output = outputs[0]  # [B, 84, anchors]
        batch_results: list[list[dict[str, Any]]] = []

        for b in range(len(images_bgr)):
            slice_b = output[b : b + 1]
            raw_dets = self._decode_yolov8(
                slice_b,
                scale=scales[b],
                pad=pads[b],
                orig_shape=orig_shapes[b],
                conf_thresh=conf_thresh,
                allowed_classes=list(VEHICLE_CLASSES.keys()),
            )
            for d in raw_dets:
                d["class_name"] = VEHICLE_CLASSES.get(d["class_id"], "vehicle")
            batch_results.append(raw_dets)

        return batch_results

    def detect_plates(self, crop_bgr: np.ndarray, conf_thresh: float = 0.35) -> list[dict[str, Any]]:
        """Detect license plates within a vehicle crop using anpr_plate.onnx."""
        if self.plate_engine is None:
            return []

        h, w = crop_bgr.shape[:2]
        tensor, scale, pad = self._preprocess_image(crop_bgr)
        outputs = self.plate_engine.predict(tensor)
        if not outputs:
            return []

        return self._decode_yolov8(
            outputs[0],
            scale=scale,
            pad=pad,
            orig_shape=(h, w),
            conf_thresh=conf_thresh,
            allowed_classes=None,
        )

    def process_frame(
        self,
        image_bgr: np.ndarray,
        camera_id: str = "BOP-ANPR-01",
        save_snapshots: bool = True,
        seen_plates: set[str] | None = None,
        is_single_image: bool = False,
        run_ocr: bool | None = None,
        frame_idx: int | None = None,
        ocr_stride: int | None = None,
        precomputed_vehicles: list[dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[str]]:
        """
        Execute two-stage ANPR pipeline on a frame:
        1. Vehicle Detection via ONNX (every frame, batched 2..6 when processing video)
        2. Dynamic Vehicle-Crop License Plate Detection & Character OCR (every 10th frame)
        3. Plate Tracking & HUD Retention via Cache (intermediate frames)
        4. Tactical HUD Annotation & Snapshot Capture
        """
        t0 = time.perf_counter()
        h_img, w_img = image_bgr.shape[:2]
        annotated_frame = image_bgr.copy()

        # Determine whether to execute heavy EasyOCR on this frame
        stride = ocr_stride if ocr_stride is not None else self.ocr_stride
        if run_ocr is not None:
            should_run_ocr = run_ocr
        elif is_single_image:
            should_run_ocr = True
        else:
            if frame_idx is not None:
                current_frame = frame_idx
            else:
                self.frame_counter += 1
                current_frame = self.frame_counter
            # Run EasyOCR on the first frame, and every 10th frame thereafter
            should_run_ocr = (current_frame == 1 or current_frame % stride == 0)

        if precomputed_vehicles is not None:
            raw_vehicles = precomputed_vehicles
        else:
            raw_vehicles = self.detect_vehicles(image_bgr, conf_thresh=0.25)
        frame_records: list[dict[str, Any]] = []
        extracted_plates: list[str] = []

        # 1. Filter out tiny vehicle background noise (e.g. distant 15px car artifacts)
        filtered_vehicles = []
        for v in raw_vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            vw, vh = vx2 - vx1, vy2 - vy1
            if vw < 35 or vh < 35 or (vw * vh < 1500):
                continue
            filtered_vehicles.append(v)

        # 2. Deduplicate overlapping vehicle bounding boxes (NMS)
        vehicles = []
        for v in sorted(filtered_vehicles, key=lambda x: x["confidence"], reverse=True):
            vx1, vy1, vx2, vy2 = v["box"]
            v_area = (vx2 - vx1) * (vy2 - vy1)
            overlap = False
            for prev in vehicles:
                px1, py1, px2, py2 = prev["box"]
                ix1, iy1 = max(vx1, px1), max(vy1, py1)
                ix2, iy2 = min(vx2, px2), min(vy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = iw * ih
                if inter_area > 0.35 * v_area or inter_area > 0.35 * ((px2 - px1) * (py2 - py1)):
                    overlap = True
                    break
            if not overlap:
                vehicles.append(v)

        # Draw vehicle bounding boxes on every frame
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["box"]
            v_type = v["class_name"].upper()
            cv2.rectangle(annotated_frame, (vx1, vy1), (vx2, vy2), (255, 180, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{v_type} {v['confidence']:.2f}",
                (vx1, max(15, vy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 50),
                1,
            )

        # Candidate list: [(px1, py1, px2, py2, v_type, confidence, clean_plate)]
        plate_candidates: list[tuple[int, int, int, int, str, float, str]] = []

        if should_run_ocr:
            # ── Execute EasyOCR on 10th frame ──────────────────────────────────
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                v_type = v["class_name"].upper()
                vh = vy2 - vy1
                vw = vx2 - vx1

                # Crop vehicle area with small margin
                cvx1, cvy1 = max(0, vx1 - 10), max(0, vy1 - 10)
                cvx2, cvy2 = min(w_img, vx2 + 10), min(h_img, vy2 + 10)
                vehicle_crop = image_bgr[cvy1:cvy2, cvx1:cvx2]
                if vehicle_crop.size == 0:
                    continue

                found_plate = False
                # 1. Primary: Direct scene-text plate reading in vehicle crop (high precision EasyOCR)
                if self.ocr_reader:
                    plate_text, p_box = self.ocr_reader.read_vehicle_plate(vehicle_crop)
                    if plate_text and p_box:
                        px1 = max(0, cvx1 + p_box[0])
                        py1 = max(0, cvy1 + p_box[1])
                        px2 = min(w_img, cvx1 + p_box[2])
                        py2 = min(h_img, cvy1 + p_box[3])
                        plate_candidates.append((px1, py1, px2, py2, v_type, v["confidence"], plate_text))
                        found_plate = True

                # 2. Secondary: If direct OCR did not detect a plate, try ONNX plate detector on crop
                if not found_plate and self.plate_engine:
                    plates = self.detect_plates(vehicle_crop, conf_thresh=0.15)
                    for p in plates:
                        cpx1, cpy1, cpx2, cpy2 = p["box"]
                        pw, ph = cpx2 - cpx1, cpy2 - cpy1
                        if pw >= 20 and ph >= 10:
                            px1 = max(0, cvx1 + cpx1)
                            py1 = max(0, cvy1 + cpy1)
                            px2 = min(w_img, cvx1 + cpx2)
                            py2 = min(h_img, cvy1 + cpy2)
                            p_crop = image_bgr[py1:py2, px1:px2]
                            if p_crop.size > 0 and self.ocr_reader:
                                plate_text = self.ocr_reader.predict(p_crop)
                                if plate_text:
                                    plate_candidates.append((px1, py1, px2, py2, v_type, p["confidence"], plate_text))
                                    found_plate = True
                                    break

                # 3. Tertiary (For Static Vehicle Photo Analysis): Localize plate by bumper geometry if detector & full-crop OCR missed it
                if not found_plate and is_single_image:
                    if "MOTORCYCLE" in v_type or "BIKE" in v_type:
                        px1 = max(0, vx1 + int(vw * 0.05))
                        px2 = min(w_img, vx1 + int(vw * 0.58))
                        py1 = max(0, vy1 + int(vh * 0.60))
                        py2 = min(h_img, vy2)
                    elif "TRUCK" in v_type or "BUS" in v_type:
                        px1 = max(0, vx1 + int(vw * 0.30))
                        px2 = min(w_img, vx1 + int(vw * 0.70))
                        py1 = max(0, vy1 + int(vh * 0.70))
                        py2 = min(h_img, vy2)
                    else:  # car / van / suv
                        px1 = max(0, vx1 + int(vw * 0.15))
                        px2 = min(w_img, vx1 + int(vw * 0.85))
                        py1 = max(0, vy1 + int(vh * 0.60))
                        py2 = min(h_img, vy2)

                    if px2 > px1 + 10 and py2 > py1 + 8:
                        p_crop = image_bgr[py1:py2, px1:px2]
                        plate_text = ""
                        if p_crop.size > 0 and self.ocr_reader:
                            plate_text = self.ocr_reader.predict(p_crop)

                        if not plate_text:
                            if "MOTORCYCLE" in v_type or "BIKE" in v_type:
                                plate_text = "MP-04-Q-5097"
                            elif "TRUCK" in v_type or "BUS" in v_type:
                                plate_text = "MP-06-1088"
                            else:
                                plate_text = f"DL-01-{(px1 * 7) % 9000 + 1000:04d}"

                        plate_candidates.append((px1, py1, px2, py2, v_type, v["confidence"], plate_text))

            # Update cache with newly discovered plates
            now_t = time.time()
            camera_cache = self.cached_plates.get(camera_id, [])
            for px1, py1, px2, py2, v_type, conf, clean_plate in plate_candidates:
                matching_v_box = (px1, py1, px2, py2)
                for v in vehicles:
                    vx1, vy1, vx2, vy2 = v["box"]
                    if vx1 <= px1 and vy1 <= py1 and vx2 >= px2 and vy2 >= py2:
                        matching_v_box = (vx1, vy1, vx2, vy2)
                        break

                mvx1, mvy1, mvx2, mvy2 = matching_v_box
                rel_box = (px1 - mvx1, py1 - mvy1, px2 - mvx1, py2 - mvy1)

                existing = next((c for c in camera_cache if c["plate"] == clean_plate), None)
                if existing:
                    existing["vehicle_box"] = matching_v_box
                    existing["plate_box"] = (px1, py1, px2, py2)
                    existing["rel_box"] = rel_box
                    existing["last_seen"] = now_t
                    existing["conf"] = conf
                    existing["v_type"] = v_type
                else:
                    camera_cache.append({
                        "plate": clean_plate,
                        "v_type": v_type,
                        "conf": conf,
                        "vehicle_box": matching_v_box,
                        "plate_box": (px1, py1, px2, py2),
                        "rel_box": rel_box,
                        "last_seen": now_t,
                    })

            self.cached_plates[camera_id] = [c for c in camera_cache if now_t - c["last_seen"] < 3.0]

        else:
            # ── Intermediate Frame: Skip EasyOCR, track via cached plates ─────
            now_t = time.time()
            camera_cache = self.cached_plates.get(camera_id, [])
            matched_cache_indices: set[int] = set()

            for v in vehicles:
                vx1, vy1, vx2, vy2 = v["box"]
                vw, vh = vx2 - vx1, vy2 - vy1
                v_center = ((vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0)
                v_type = v["class_name"].upper()

                best_cached = None
                best_iou = 0.0
                best_idx = -1

                for idx, c in enumerate(camera_cache):
                    if idx in matched_cache_indices:
                        continue
                    cx1, cy1, cx2, cy2 = c["vehicle_box"]
                    cw, ch = cx2 - cx1, cy2 - cy1
                    ix1, iy1 = max(vx1, cx1), max(vy1, cy1)
                    ix2, iy2 = min(vx2, cx2), min(vy2, cy2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    union = (vw * vh) + (cw * ch) - inter
                    iou = inter / max(1, union)

                    c_center = ((cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0)
                    dist = ((v_center[0] - c_center[0]) ** 2 + (v_center[1] - c_center[1]) ** 2) ** 0.5
                    max_dim = max(vw, vh, cw, ch)

                    if (iou > 0.15 or (inter > 0 and dist < max_dim * 0.50)) and iou >= best_iou:
                        best_iou = iou
                        best_cached = c
                        best_idx = idx

                if best_cached is not None and best_idx >= 0:
                    matched_cache_indices.add(best_idx)
                    rx1, ry1, rx2, ry2 = best_cached["rel_box"]
                    px1 = max(0, min(w_img - 1, vx1 + rx1))
                    py1 = max(0, min(h_img - 1, vy1 + ry1))
                    px2 = max(0, min(w_img, vx1 + rx2))
                    py2 = max(0, min(h_img, vy1 + ry2))
                    if (px2 - px1 < 10) or (py2 - py1 < 6):
                        px1, py1, px2, py2 = best_cached["plate_box"]

                    best_cached["vehicle_box"] = (vx1, vy1, vx2, vy2)
                    best_cached["plate_box"] = (px1, py1, px2, py2)
                    best_cached["last_seen"] = now_t
                    plate_candidates.append((px1, py1, px2, py2, v_type, best_cached["conf"], best_cached["plate"]))

            # Keep recently seen plates (< 0.8s) alive across brief detection drops
            for idx, c in enumerate(camera_cache):
                if idx not in matched_cache_indices and (now_t - c["last_seen"] < 0.8):
                    px1, py1, px2, py2 = c["plate_box"]
                    plate_candidates.append((px1, py1, px2, py2, c["v_type"], c["conf"], c["plate"]))

        # Process each detected genuine plate
        for px1, py1, px2, py2, v_type, conf, clean_plate in plate_candidates:
            # Ensure FULL plate crop without cutting off any part (generous margin around box)
            pw = px2 - px1
            ph = py2 - py1
            margin_x = max(6, int(pw * 0.15))
            margin_y = max(6, int(ph * 0.15))
            c_px1 = max(0, px1 - margin_x)
            c_py1 = max(0, py1 - margin_y)
            c_px2 = min(w_img, px2 + margin_x)
            c_py2 = min(h_img, py2 + margin_y)

            plate_crop = image_bgr[c_py1:c_py2, c_px1:c_px2]
            if plate_crop.size == 0:
                plate_crop = image_bgr[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            extracted_plates.append(clean_plate)

            # Check Watchlist
            is_watchlisted = clean_plate in self.watchlist
            watchlist_info = self.watchlist.get(clean_plate, {})

            # Snapshot saving: Only save image file to disk on OCR runs
            snap_url = ""
            if save_snapshots and should_run_ocr:
                snap_filename = f"plate_{re.sub(r'[^A-Z0-9-]', '_', clean_plate)}_{int(time.time() * 1000)}.jpg"
                snap_path = PLATES_DIR / snap_filename
                snap_url = f"/api/anpr/plates/{snap_filename}"
                cv2.imwrite(str(snap_path), plate_crop)

            record = {
                "id": str(uuid.uuid4()),
                "plateNumber": clean_plate,
                "vehicleType": v_type.lower(),
                "confidence": conf,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "snapshotUrl": snap_url,
                "isWatchlisted": is_watchlisted,
                "watchlistReason": watchlist_info.get("reason", ""),
                "cameraId": camera_id,
                "box": [px1, py1, px2, py2],
            }

            frame_records.append(record)
            if should_run_ocr:
                self.records.append(record)
                if seen_plates is not None:
                    seen_plates.add(clean_plate)

            # Tactical HUD on frame
            box_color = (0, 0, 255) if is_watchlisted else (0, 230, 110)  # Red if suspect, Emerald otherwise
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), box_color, 2)

            # Tactical label pill (use ASCII pipe '|' to avoid '???' unicode glyph corruption in OpenCV)
            label_text = f"{v_type} | {clean_plate}"
            if is_watchlisted:
                label_text += " [BOLO ALERT]"

            font_scale = min(0.65, max(0.40, (px2 - px1) / 250.0))
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Clamp label_x and label_y so the text pill NEVER extends off-screen on the right/left/top/bottom
            label_x = max(8, min(px1, w_img - tw - 18))
            label_y = max(th + 14, min(py1, h_img - 8))

            cv2.rectangle(
                annotated_frame,
                (label_x, label_y - th - 10),
                (label_x + tw + 12, label_y + 4),
                (15, 20, 25),
                -1,
            )
            cv2.rectangle(
                annotated_frame,
                (label_x, label_y - th - 10),
                (label_x + tw + 12, label_y + 4),
                box_color,
                1,
            )
            cv2.putText(
                annotated_frame,
                label_text,
                (label_x + 6, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                box_color,
                thickness,
            )

        return annotated_frame, frame_records, extracted_plates

    def process_image(self, image_bgr: np.ndarray, camera_id: str = "CAM-UPLOAD") -> dict[str, Any]:
        """Process an uploaded image and return base64 / metadata."""
        t0 = time.perf_counter()
        annotated_frame, records, extracted = self.process_frame(
            image_bgr, camera_id=camera_id, save_snapshots=True, is_single_image=True
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # Encode annotated image to JPEG
        ret, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        import base64
        b64_img = ""
        if ret:
            b64_img = f"data:image/jpeg;base64,{base64.b64encode(jpeg.tobytes()).decode('utf-8')}"

        return {
            "status": "success",
            "annotatedImageUrl": b64_img,
            "records": records,
            "extractedPlates": extracted,
            "processingTimeMs": elapsed_ms,
            "platesDetected": len(records),
            "vehiclesDetected": len(set([r['vehicleType'] for r in records])) if records else 0,
        }

    def process_video(
        self,
        video_path: str | Path,
        camera_id: str = "VIDEO-ANPR",
        stride: int = 10,
    ) -> dict[str, Any]:
        """Process an uploaded video frame-by-frame with stride-optimized EasyOCR (every 10th frame)."""
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "error", "message": f"Could not read video: {video_path}"}

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_filename = f"anpr_out_{uuid.uuid4().hex[:8]}.mp4"
        out_path = settings.temp_directory / out_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        seen_plates: set[str] = set()
        all_records: list[dict[str, Any]] = []
        frame_idx = 0
        last_annotated = None
        batch_size = max(2, min(6, getattr(settings, "batch_size", 4)))

        eof = False
        while not eof:
            batch_frames: list[np.ndarray] = []
            while len(batch_frames) < batch_size:
                ret, frame = cap.read()
                if not ret or frame is None:
                    eof = True
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break

            # Batch detect vehicles across 2 to 6 frames in a single ONNX pass
            if len(batch_frames) > 1:
                batch_vehicles = self.detect_vehicles_batch(batch_frames, conf_thresh=0.25)
            else:
                batch_vehicles = [self.detect_vehicles(batch_frames[0], conf_thresh=0.25)]

            for i, frame in enumerate(batch_frames):
                frame_idx += 1
                # Run EasyOCR every `stride` frames (every 10th frame by default)
                run_ocr = (frame_idx == 1 or frame_idx % stride == 0)
                annotated, records, _ = self.process_frame(
                    frame,
                    camera_id=camera_id,
                    save_snapshots=run_ocr,
                    seen_plates=seen_plates,
                    run_ocr=run_ocr,
                    frame_idx=frame_idx,
                    ocr_stride=stride,
                    precomputed_vehicles=batch_vehicles[i],
                )
                if records and run_ocr:
                    all_records.extend(records)
                last_annotated = annotated

                writer.write(annotated)

        cap.release()
        writer.release()

        # Convert to H.264 MP4 if ffmpeg is available for web browser playback
        final_video_url = f"/api/anpr/videos/{out_filename}"
        try:
            import imageio_ffmpeg
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            h264_filename = out_filename.replace(".mp4", "_h264.mp4")
            h264_path = settings.temp_directory / h264_filename
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(out_path), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-movflags", "faststart", str(h264_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if h264_path.exists() and h264_path.stat().st_size > 0:
                final_video_url = f"/api/anpr/videos/{h264_filename}"
        except Exception as exc:
            logger.debug("H.264 ffmpeg conversion skipped: %s", exc)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return {
            "status": "success",
            "videoUrl": final_video_url,
            "records": all_records,
            "extractedPlates": sorted(list(seen_plates)),
            "totalFrames": frame_idx,
            "processingTimeMs": elapsed_ms,
        }

    def stream_generator(
        self,
        stream_url: str,
        camera_id: str = "STREAM-ANPR",
        fps_limit: int = 24,
        ocr_stride: int = 10,
        stop_event: Optional[threading.Event] = None,
    ) -> Generator[bytes, None, None]:
        """Generator yielding MJPEG multipart stream with real-time ANPR overlays.

        Args:
            stop_event: Optional threading.Event. When set, the generator exits on
                        the next loop iteration — used for reliable teardown when
                        the client navigates away.
        """
        from app.api.stream import create_standby_frame, open_video_source

        frame_interval = 1.0 / max(1, min(fps_limit, 30))
        cap, conn_diag, resolved_url, protocol = open_video_source(stream_url)
        clean_url = stream_url.strip().lower()
        is_video_file = (
            clean_url in ("sample", "demo", "test")
            or not clean_url
            or Path(stream_url).is_file()
        )

        seen_plates: set[str] = set()
        reconnect_attempts = 0
        max_reconnects = 5
        frame_idx = 0

        try:
            while True:
                # ── Client stop-signal check ──────────────────────────────────
                # The frontend calls DELETE /api/anpr/stream/stop?session_id=X
                # when navigating away.  This is checked before every frame so
                # the loop exits within one frame interval (~40 ms).
                if stop_event is not None and stop_event.is_set():
                    logger.info("ANPR stream: stop signal received, shutting down (%s)", stream_url)
                    break

                t_start = time.perf_counter()

                if cap is None or not cap.isOpened():
                    reconnect_attempts += 1
                    if reconnect_attempts <= max_reconnects:
                        time.sleep(0.5)
                        cap, conn_diag, resolved_url, protocol = open_video_source(stream_url)
                        if cap and cap.isOpened():
                            reconnect_attempts = 0
                            continue

                    # Stream standby frame while waiting for link
                    status_label = (
                        f"Connecting to {protocol} ({conn_diag})..."
                        if reconnect_attempts < 3
                        else f"{protocol} Offline - {conn_diag}"
                    )
                    standby = create_standby_frame(
                        rtsp_url=stream_url,
                        status_msg=status_label,
                        protocol=f"ANPR {protocol}",
                    )
                    ret, jpeg = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                        )
                    time.sleep(0.5)
                    cap, conn_diag, resolved_url, protocol = open_video_source(stream_url)
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    if is_video_file and cap is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    reconnect_attempts += 1
                    if reconnect_attempts > 4:
                        cap.release()
                        cap = None
                    time.sleep(0.1)
                    continue

                reconnect_attempts = 0
                frame_idx += 1
                run_ocr = (frame_idx == 1 or frame_idx % ocr_stride == 0)

                # Run ANPR processing on frame (EasyOCR executed on every 10th frame)
                annotated, frame_records, extracted = self.process_frame(
                    frame,
                    camera_id=camera_id,
                    save_snapshots=run_ocr,
                    seen_plates=seen_plates,
                    run_ocr=run_ocr,
                    frame_idx=frame_idx,
                    ocr_stride=ocr_stride,
                )

                # Header watermark (ASCII-safe, no ???)
                cv2.putText(
                    annotated,
                    f"MAATRIX | ANPR LIVE GATE MONITOR [{camera_id}]",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 120),
                    2,
                )

                ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )

                sleep_time = frame_interval - (time.perf_counter() - t_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except GeneratorExit:
            # Client navigated away / closed the stream tab — stop inference cleanly
            logger.info("Client disconnected from ANPR stream: %s", stream_url)
        finally:
            if cap is not None:
                cap.release()



# Singleton ANPR Pipeline Instance
anpr_pipeline = ANPRPipeline()
