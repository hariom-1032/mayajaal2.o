"""
backend/scripts/quantize_model.py
Quantizes FP32 ONNX models in backend/models to INT8 for 2-3x faster CPU execution.
"""
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def quantize_all():
    models_to_quantize = [
        "face_recognition_sface.onnx",
        "vehicle_detector.onnx",
    ]

    for model_name in models_to_quantize:
        src = MODELS_DIR / model_name
        dst = MODELS_DIR / f"{src.stem}_int8.onnx"

        if not src.exists():
            print(f"Skipping {model_name} (file not found in {MODELS_DIR})")
            continue

        print(f"Quantizing {model_name} -> {dst.name}...")
        try:
            quantize_dynamic(
                model_input=str(src),
                model_output=str(dst),
                weight_type=QuantType.QUInt8,
            )
            print(
                f"Successfully quantized {model_name}! "
                f"Original: {src.stat().st_size / 1e6:.1f} MB -> "
                f"INT8: {dst.stat().st_size / 1e6:.1f} MB"
            )
        except Exception as exc:
            print(f"Failed to quantize {model_name}: {exc}")


if __name__ == "__main__":
    quantize_all()
