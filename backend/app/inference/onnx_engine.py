from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Ensure CUDA and cuDNN DLLs are discoverable on Windows
from app.utils.cuda_setup import (
    setup_cuda_dll_directories,
    get_cuda_device_count,
    get_cuda_device_name,
)
setup_cuda_dll_directories()

import numpy as np
import onnxruntime as ort


from app.core.config import settings


logger = logging.getLogger(__name__)


class ONNXEngineError(Exception):
    """Base exception for ONNX engine errors."""


class ONNXModelError(ONNXEngineError):
    """Raised when an ONNX model cannot be loaded or is invalid."""


class ONNXInferenceError(ONNXEngineError):
    """Raised when inference fails."""


class ONNXEngine:
    """
    Safe, reusable ONNX Runtime inference engine.

    Designed for:
    - GPU acceleration (CUDAExecutionProvider) when available
    - CPU fallback (CPUExecutionProvider)
    - Multiple ONNX models
    - Loading models only once
    - Controlled CPU usage
    - Dynamic model input/output inspection
    """

    def __init__(
        self,
        model_path: str | Path,
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
        gpu_mem_limit_gb: float | None = None,
    ) -> None:
        # gpu_mem_limit_gb: VRAM cap per engine.
        # None / 0 → read from settings (0 in settings = no hard cap).
        resolved_limit = gpu_mem_limit_gb if gpu_mem_limit_gb is not None else getattr(settings, "gpu_mem_limit_gb", 0.0)
        self.gpu_mem_limit_bytes: int | None = int(resolved_limit * 1024 ** 3) if resolved_limit > 0 else None
        self.model_path = Path(model_path)

        self.intra_op_threads = max(1, intra_op_threads)
        self.inter_op_threads = max(1, inter_op_threads)

        self.session: ort.InferenceSession | None = None
        self._use_cuda: bool = False  # set after session creation
        self.metadata: dict[str, str] = {}
        self.class_labels: list[str] | None = None
        self.input_size: tuple[int, int] | None = None

        self._validate_model_path()
        self._create_session()

    def _validate_model_path(self) -> None:
        """Validate the ONNX model path before loading."""

        if not self.model_path.exists():
            raise ONNXModelError(
                f"ONNX model not found: {self.model_path}"
            )

        if not self.model_path.is_file():
            raise ONNXModelError(
                f"Model path is not a file: {self.model_path}"
            )

        if self.model_path.suffix.lower() != ".onnx":
            raise ONNXModelError(
                f"Expected an .onnx file, got: {self.model_path.name}"
            )

        if self.model_path.stat().st_size == 0:
            raise ONNXModelError(
                f"ONNX model is empty: {self.model_path}"
            )

    def _get_execution_providers(self) -> list:
        """Detect and return preferred execution providers with tuned options.

        Returns a list of (provider_name, options_dict) tuples so that
        CUDAExecutionProvider uses the arena allocator, best cuDNN conv algo,
        and a configurable VRAM cap.
        """
        available = ort.get_available_providers()
        requested_device = getattr(settings, "device", "auto").lower().strip()

        providers: list = []

        if requested_device in ("auto", "cuda", "gpu"):
            if "CUDAExecutionProvider" in available:
                cuda_count = get_cuda_device_count()
                req_id = getattr(settings, "cuda_device_id", 0)
                # Map device_id safely: CUDA only indexes NVIDIA devices (starting at 0).
                # If user specifies device 1 (referencing Windows Task Manager 'GPU 1'),
                # map to device 0 if only 1 NVIDIA GPU exists.
                if cuda_count > 0 and req_id >= cuda_count:
                    resolved_device_id = 0
                    device_name = get_cuda_device_name(0)
                    logger.info(
                        "Requested CUDA device_id=%d mapped to device 0: %s (1 NVIDIA GPU present)",
                        req_id,
                        device_name,
                    )
                else:
                    resolved_device_id = req_id
                    device_name = get_cuda_device_name(resolved_device_id)

                cuda_options: dict = {
                    "device_id": resolved_device_id,
                    # kNextPowerOfTwo reduces allocator fragmentation on variable batch sizes
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    # EXHAUSTIVE search finds the fastest cuDNN conv algorithm per shape
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    # Run copies in the default CUDA stream to overlap with compute
                    "do_copy_in_default_stream": True,
                }
                if self.gpu_mem_limit_bytes is not None:
                    cuda_options["gpu_mem_limit"] = self.gpu_mem_limit_bytes
                    _limit_log = f"{self.gpu_mem_limit_bytes / 1024 ** 3:.1f} GB"
                else:
                    _limit_log = "unlimited"
                providers.append(("CUDAExecutionProvider", cuda_options))
                logger.info(
                    "CUDAExecutionProvider selected | device_id=%d (%s) | gpu_mem_limit=%s | cudnn_conv_algo=EXHAUSTIVE",
                    resolved_device_id,
                    device_name,
                    _limit_log,
                )
            elif requested_device in ("cuda", "gpu"):
                logger.warning(
                    "CUDA/GPU device requested, but 'CUDAExecutionProvider' is not in available providers %s. Falling back to CPU.",
                    available,
                )

        providers.append(("CPUExecutionProvider", {}))
        return providers

    def _create_session(self) -> None:
        """Create the ONNX Runtime session with all performance optimizations."""

        try:
            session_options = ort.SessionOptions()

            # ── Graph optimizations ──────────────────────────────────────────
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            # Persist the fused/optimized graph next to the model file so that
            # subsequent restarts skip the optimization pass entirely.
            optimized_path = self.model_path.with_suffix(".optimized.onnx")
            session_options.optimized_model_filepath = str(optimized_path)

            # ── Execution mode ───────────────────────────────────────────────
            # ORT_PARALLEL allows independent graph nodes to run concurrently.
            # Tune SIH_INTRA_OP_THREADS (= physical cores) and
            # SIH_INTER_OP_THREADS via .env for best CPU throughput.
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            # ── CPU thread control ───────────────────────────────────────────
            session_options.intra_op_num_threads = self.intra_op_threads
            session_options.inter_op_num_threads = self.inter_op_threads

            # ── Memory optimizations ─────────────────────────────────────────
            # Reuse memory allocations across inference calls to reduce GC pressure.
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True
            session_options.enable_cpu_mem_arena = True

            # ── Execution providers (GPU preferred, CPU fallback) ─────────────
            providers = self._get_execution_providers()

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers,
            )

            active_providers = self.session.get_providers()
            self._use_cuda = "CUDAExecutionProvider" in active_providers

            self._extract_metadata()

            logger.info(
                "Loaded ONNX model: %s | Providers: %s | CUDA IO Binding: %s | "
                "Class labels: %s | Input size: %s | Optimized graph: %s",
                self.model_path.name,
                active_providers,
                self._use_cuda,
                len(self.class_labels) if self.class_labels else "none",
                self.input_size or "dynamic",
                optimized_path.name,
            )

        except Exception as exc:
            logger.exception(
                "Failed to load ONNX model: %s",
                self.model_path,
            )

            raise ONNXModelError(
                f"Failed to load ONNX model '{self.model_path.name}': {exc}"
            ) from exc

    def _extract_metadata(self) -> None:
        """Extract embedded model metadata, class labels, and input image size."""
        if self.session is None:
            return

        try:
            model_meta = self.session.get_modelmeta()
            if model_meta and model_meta.custom_metadata_map:
                self.metadata = dict(model_meta.custom_metadata_map)

                # 1. Parse class names/labels if embedded in metadata
                if "names" in self.metadata:
                    raw_names = self.metadata["names"]
                    parsed = None
                    try:
                        import ast
                        parsed = ast.literal_eval(raw_names)
                    except Exception:
                        try:
                            import json
                            parsed = json.loads(raw_names)
                        except Exception:
                            pass

                    if isinstance(parsed, dict):
                        # Sort by integer keys (e.g. {0: 'Blunt_Weapon', ...})
                        sorted_items = sorted(
                            [(int(k), str(v)) for k, v in parsed.items()],
                            key=lambda x: x[0],
                        )
                        self.class_labels = [v for _, v in sorted_items]
                    elif isinstance(parsed, list):
                        self.class_labels = [str(x) for x in parsed]

                # 2. Parse imgsz (e.g. [640, 640])
                if "imgsz" in self.metadata:
                    raw_imgsz = self.metadata["imgsz"]
                    try:
                        import ast
                        val = ast.literal_eval(raw_imgsz)
                        if isinstance(val, (list, tuple)) and len(val) >= 2:
                            self.input_size = (int(val[1]), int(val[0]))  # (width, height)
                        elif isinstance(val, int):
                            self.input_size = (val, val)
                    except Exception:
                        pass

        except Exception as exc:
            logger.debug(
                "Failed to extract metadata from model '%s': %s",
                self.model_path.name,
                exc,
            )

        # Fallback: check static input shape from session
        if not self.input_size and self.session.get_inputs():
            first_input = self.session.get_inputs()[0]
            shape = first_input.shape
            if len(shape) == 4:
                # Shape is [B, C, H, W]
                h, w = shape[2], shape[3]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    self.input_size = (w, h)

    def get_inputs(self) -> list[dict[str, Any]]:
        """Return information about model inputs."""

        if self.session is None:
            raise ONNXModelError("ONNX session is not initialized.")

        inputs = []

        for input_meta in self.session.get_inputs():
            inputs.append(
                {
                    "name": input_meta.name,
                    "shape": input_meta.shape,
                    "type": input_meta.type,
                }
            )

        return inputs

    def get_outputs(self) -> list[dict[str, Any]]:
        """Return information about model outputs."""

        if self.session is None:
            raise ONNXModelError("ONNX session is not initialized.")

        outputs = []

        for output_meta in self.session.get_outputs():
            outputs.append(
                {
                    "name": output_meta.name,
                    "shape": output_meta.shape,
                    "type": output_meta.type,
                }
            )

        return outputs

    def predict(
        self,
        input_data: np.ndarray,
        input_name: str | None = None,
    ) -> list[np.ndarray]:
        """
        Run inference using a NumPy array.

        When CUDAExecutionProvider is active, IO Binding is used to keep
        tensors on-device and avoid redundant CPU<->GPU copies, which
        significantly reduces latency on large batch inputs.

        Preprocessing is intentionally kept outside this class because
        Riyan's final ONNX models are not available yet.
        """

        if self.session is None:
            raise ONNXInferenceError(
                "Cannot run inference: ONNX session is not initialized."
            )

        if not isinstance(input_data, np.ndarray):
            raise ONNXInferenceError(
                "input_data must be a NumPy ndarray."
            )

        if input_data.size == 0:
            raise ONNXInferenceError(
                "input_data cannot be empty."
            )

        try:
            if input_name is None:
                model_inputs = self.session.get_inputs()

                if not model_inputs:
                    raise ONNXInferenceError(
                        "The ONNX model has no inputs."
                    )

                input_name = model_inputs[0].name

            # ── GPU path: IO Binding avoids CPU<->GPU memcpy overhead ─────────
            if self._use_cuda:
                io_binding = self.session.io_binding()

                # Guard against CUDA error 700 (illegal memory access):
                # OrtValue.ortvalue_from_numpy reads raw buffer pointers, so
                # the array MUST be C-contiguous and the correct dtype (float32).
                # Non-contiguous arrays (from slicing/transposing) or float64
                # arrays will cause cudaMemcpy(HostToDevice) to access invalid
                # memory addresses.
                expected_dtype = np.float32
                if not input_data.flags["C_CONTIGUOUS"] or input_data.dtype != expected_dtype:
                    input_data = np.ascontiguousarray(input_data, dtype=expected_dtype)

                # Bind input: place tensor directly on CUDA device
                input_ortvalue = ort.OrtValue.ortvalue_from_numpy(
                    input_data, device_type="cuda", device_id=0
                )
                io_binding.bind_input(
                    name=input_name,
                    device_type="cuda",
                    device_id=0,
                    element_type=input_data.dtype,
                    shape=input_data.shape,
                    buffer_ptr=input_ortvalue.data_ptr(),
                )

                # Bind all outputs on the CUDA device
                for out_meta in self.session.get_outputs():
                    io_binding.bind_output(out_meta.name, device_type="cuda")

                self.session.run_with_iobinding(io_binding)

                # Transfer outputs back to CPU NumPy arrays
                return [ortval.numpy() for ortval in io_binding.get_outputs()]

            # ── CPU path: standard run ────────────────────────────────────────
            return self.session.run(None, {input_name: input_data})

        except ONNXInferenceError:
            raise

        except Exception as exc:
            logger.exception(
                "ONNX inference failed for model: %s",
                self.model_path.name,
            )

            raise ONNXInferenceError(
                f"Inference failed for '{self.model_path.name}': {exc}"
            ) from exc

    def get_model_info(self) -> dict[str, Any]:
        """Return useful information about the loaded model."""

        if self.session is None:
            raise ONNXModelError("ONNX session is not initialized.")

        return {
            "model_name": self.model_path.name,
            "model_path": str(self.model_path),
            "providers": self.session.get_providers(),
            "cuda_io_binding": self._use_cuda,
            "gpu_mem_limit_gb": (
                round(self.gpu_mem_limit_bytes / 1024 ** 3, 2)
                if self.gpu_mem_limit_bytes is not None
                else "unlimited"
            ),
            "inputs": self.get_inputs(),
            "outputs": self.get_outputs(),
            "class_labels": self.class_labels,
            "input_size": list(self.input_size) if self.input_size else None,
            "task": self.metadata.get("task", "detect"),
            "intra_op_threads": self.intra_op_threads,
            "inter_op_threads": self.inter_op_threads,
        }

    def close(self) -> None:
        """Release the ONNX Runtime session."""

        if self.session is not None:
            self.session = None

            logger.info(
                "Closed ONNX model: %s",
                self.model_path.name,
            )

    def __enter__(self) -> "ONNXEngine":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()