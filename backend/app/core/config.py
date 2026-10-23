from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """
    Central configuration for the SIH26187 inference backend.
    """

    # Application
    app_name: str = "SIH26187 Intelligent Video Analytics Backend"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model configuration
    model_directory: Path = BASE_DIR / "models"
    max_models: int = Field(default=4, ge=1, le=4)
    device: str = Field(default="auto", description="Execution device: 'auto', 'cuda', or 'cpu'")
    cuda_device_id: int = Field(default=0, ge=0, description="CUDA device ID (0 for first NVIDIA GPU).")
    batch_size: int = Field(default=4, ge=2, le=6, description="Default ONNX batch inference size (2 to 6 frames at once).")

    # ONNX Runtime CPU configuration
    intra_op_threads: int = Field(default=2, ge=1)
    inter_op_threads: int = Field(default=1, ge=1)

    # GPU memory cap per ONNX engine (0 = no hard cap, use all available VRAM)
    gpu_mem_limit_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Max VRAM (GB) each ONNX CUDA arena may allocate. 0 = unlimited.",
    )

    # Inference concurrency
    max_concurrent_inference: int = Field(default=2, ge=1)

    # Request/resource limits
    max_upload_size_mb: int = Field(default=200, ge=1)
    inference_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
    )

    # Supported input types
    allowed_image_extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    )

    allowed_video_extensions: tuple[str, ...] = (
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    )

    # ByteTrack multi-object tracking defaults
    # All values can be overridden via SIH_TRACKER_* environment variables.
    tracker_activation_threshold: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Min detection confidence to activate a new ByteTrack track.",
    )
    tracker_lost_track_buffer: int = Field(
        default=30,
        ge=1,
        description="Frames to hold a lost track before permanently removing it.",
    )
    tracker_frame_rate: int = Field(
        default=30,
        ge=1,
        description="Expected stream FPS used by the internal Kalman filter.",
    )
    tracker_min_consecutive_frames: int = Field(
        default=1,
        ge=1,
        description="Consecutive matched frames before a track is confirmed.",
    )

    # Logging
    log_level: str = "INFO"
    log_directory: Path = BASE_DIR / "logs"

    # Temporary files
    temp_directory: Path = BASE_DIR / "temp"

    model_config = SettingsConfigDict(
        env_prefix="SIH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()