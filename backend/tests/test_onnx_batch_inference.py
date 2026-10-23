"""
Verification test script for ONNX Batch Inference (2 to 6 frames at once).
Tests:
1. Config validation (batch_size default, range 2..6)
2. Preprocessor batch stacking for B in [2, 3, 4, 5, 6]
3. Postprocessor batch decoding (YOLOv8, End-to-End) across dynamic batches
4. ONNXEngine & InferenceService batch forward pass (B=2, B=4, B=6)
"""
import sys
import numpy as np
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.runtime import inference_service, model_manager
from app.pipeline.preprocessor import Preprocessor, PreprocessingConfig
from app.pipeline.postprocessor import Postprocessor, PostprocessorConfig

def run_tests():
    total_tests = 0
    passed_tests = 0

    print("=" * 60)
    print("RUNNING ONNX BATCH INFERENCE VERIFICATION (2 TO 6 FRAMES)")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Config settings.batch_size
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 1] Config settings.batch_size")
    print(f"   settings.batch_size: {settings.batch_size}")
    assert 2 <= settings.batch_size <= 6, f"settings.batch_size={settings.batch_size} out of range [2, 6]"
    print("   [PASS] Config batch_size within [2, 6]")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 2: Preprocessor.process_batch for B in [2, 3, 4, 5, 6]
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 2] Preprocessor.process_batch across B in [2, 3, 4, 5, 6]")
    prep_config = PreprocessingConfig(
        target_width=640,
        target_height=640,
        convert_bgr_to_rgb=True,
        normalize=False,
        scale=1.0 / 255.0,
        channel_first=True,
        add_batch_dimension=True,
    )
    preprocessor = Preprocessor(prep_config)

    for b in [2, 3, 4, 5, 6]:
        dummy_frames = [np.full((480 + i * 10, 640, 3), 100 + i * 20, dtype=np.uint8) for i in range(b)]
        batch_tensor = preprocessor.process_batch(dummy_frames)
        assert batch_tensor.shape == (b, 3, 640, 640), f"Expected shape ({b}, 3, 640, 640), got {batch_tensor.shape}"
        assert batch_tensor.dtype == np.float32, f"Expected float32, got {batch_tensor.dtype}"
        print(f"   Batch size {b}: tensor shape = {batch_tensor.shape}, dtype = {batch_tensor.dtype}")

    print("   [PASS] Preprocessor.process_batch produces correct 4D tensors for all batch sizes 2..6")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 3: Postprocessor.decode_batch for dynamic batch sizes
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 3] Postprocessor.decode_batch for dynamic batch sizes")
    post_config = PostprocessorConfig(
        conf_threshold=0.25,
        iou_threshold=0.45,
        max_detections=100,
    )
    postprocessor = Postprocessor(post_config)

    # Test End-to-End format [B, 300, 6]
    for b in [2, 4, 6]:
        dummy_e2e = np.zeros((b, 300, 6), dtype=np.float32)
        # Put 1 fake detection in frame 0 and 2 fake detections in frame 1
        dummy_e2e[0, 0] = [10, 20, 100, 150, 0.90, 0]
        if b > 1:
            dummy_e2e[1, 0] = [30, 40, 120, 200, 0.85, 1]
            dummy_e2e[1, 1] = [50, 60, 220, 300, 0.75, 2]

        orig_sizes = [(640, 480) for _ in range(b)]
        decoded = postprocessor.decode_batch(dummy_e2e, orig_sizes)
        assert len(decoded) == b, f"Expected {b} frames in output, got {len(decoded)}"
        assert len(decoded[0]) == 1, f"Expected 1 detection in frame 0, got {len(decoded[0])}"
        assert len(decoded[1]) == 2, f"Expected 2 detections in frame 1, got {len(decoded[1])}"
        print(f"   Decoded End-to-End batch {b}: {[len(d) for d in decoded]} detections per frame")

    # Test YOLOv8 format [B, 84, 8400]
    for b in [2, 4, 6]:
        dummy_v8 = np.zeros((b, 84, 8400), dtype=np.float32)
        # Frame 0: 1 detection with high score
        dummy_v8[0, 0:4, 100] = [320, 240, 100, 100]
        dummy_v8[0, 4, 100] = 0.95
        orig_sizes = [(640, 480) for _ in range(b)]
        decoded_v8 = postprocessor.decode_batch(dummy_v8, orig_sizes)
        assert len(decoded_v8) == b, f"Expected {b} frames, got {len(decoded_v8)}"
        assert len(decoded_v8[0]) >= 1, f"Expected >= 1 detection in frame 0"
        print(f"   Decoded YOLOv8 batch {b}: {[len(d) for d in decoded_v8]} detections per frame")

    print("   [PASS] Postprocessor.decode_batch correctly decodes multi-frame batches")
    passed_tests += 1

    # -------------------------------------------------------------
    # Test 4: Live ONNX Model Batch Inference via InferenceService
    # -------------------------------------------------------------
    total_tests += 1
    print("\n[Test 4] Live ONNX Model Batch Inference (vehicle_detector.onnx)")
    veh_model_path = settings.model_directory / "vehicle_detector.onnx"
    if not model_manager.is_loaded("vehicle_detector"):
        model_manager.load_model("vehicle_detector", veh_model_path)
    for b in [2, 4, 6]:
        test_frames = [np.full((480, 640, 3), 128, dtype=np.uint8) for _ in range(b)]
        batch_tensor = preprocessor.process_batch(test_frames)
        orig_sizes = [(640, 480) for _ in range(b)]

        result = inference_service.predict(
            model_name="vehicle_detector",
            input_data=batch_tensor,
            postprocess=True,
            original_image_sizes=orig_sizes,
        )
        assert result.get("status") == "success", f"Inference failed: {result}"
        assert result.get("batch_size") == b, f"Expected batch_size={b}, got {result.get('batch_size')}"
        batch_dets = result.get("batch_detections")
        assert isinstance(batch_dets, list) and len(batch_dets) == b, f"Expected {b} batch detections"
        print(f"   Batch size {b}: latency = {result.get('inference_time_ms')}ms ({round(result.get('inference_time_ms', 0) / b, 2)}ms/frame), output frames = {len(batch_dets)}")

    print("   [PASS] Live vehicle_detector.onnx batch inference succeeds for B=2, 4, 6")
    passed_tests += 1

    print("\n" + "=" * 60)
    print(f"ALL {passed_tests}/{total_tests} ONNX BATCH INFERENCE TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
