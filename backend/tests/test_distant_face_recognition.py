from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.face_service import (
    face_service,
    enhance_aligned_face,
    deduplicate_face_detections,
)


def test_enhance_aligned_face_distant():
    """Verify adaptive unsharp masking and CLAHE for low-res distant crops."""
    # Create synthetic blurry face crop
    crop = np.full((112, 112, 3), 120, dtype=np.uint8)
    cv2.circle(crop, (40, 45), 10, (50, 50, 50), -1)  # Eye 1
    cv2.circle(crop, (72, 45), 10, (50, 50, 50), -1)  # Eye 2
    cv2.line(crop, (40, 80), (72, 80), (40, 40, 40), 3)  # Mouth

    # Test standard enhancement
    std_enhanced = enhance_aligned_face(crop, is_distant_or_low_res=False)
    assert std_enhanced.shape == (112, 112, 3)

    # Test distant enhancement (with unsharp masking)
    distant_enhanced = enhance_aligned_face(crop, is_distant_or_low_res=True)
    assert distant_enhanced.shape == (112, 112, 3)
    # Check that high-frequency edges are intensified
    diff = np.abs(distant_enhanced.astype(np.float32) - crop.astype(np.float32))
    assert np.mean(diff) > 0.0
    print("[PASS] enhance_aligned_face correctly enhances distant low-res face crops.")


def test_deduplicate_face_detections():
    """Verify NMS IoU suppression merges duplicate/overlapping face detections."""
    det1 = {
        "bbox": [100.0, 100.0, 160.0, 160.0],
        "confidence": 0.85,
        "match_score": 0.62,
        "name": "Target_A",
    }
    # Highly overlapping detection (IoU ~0.8)
    det2 = {
        "bbox": [102.0, 101.0, 159.0, 161.0],
        "confidence": 0.70,
        "match_score": 0.40,
        "name": "Target_A",
    }
    # Separate distant detection
    det3 = {
        "bbox": [400.0, 400.0, 440.0, 440.0],
        "confidence": 0.80,
        "match_score": 0.55,
        "name": "Target_B",
    }

    merged = deduplicate_face_detections([det1, det2, det3], iou_thresh=0.35)
    assert len(merged) == 2, f"Expected 2 merged detections, got {len(merged)}"
    assert merged[0]["name"] == "Target_A"
    assert merged[0]["match_score"] == 0.62
    assert merged[1]["name"] == "Target_B"
    print("[PASS] deduplicate_face_detections successfully removed overlapping candidate.")


def test_person_guided_roi_and_distant_mode():
    """Verify person-guided RoI detection pipeline on synthetic surveillance frame."""
    # Create 1080p synthetic surveillance scene
    frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)

    # Simulate person bounding box for a distant person (width 50px, height 120px)
    person_box = [600, 300, 650, 420]

    # Run detect_and_recognize with distant mode and person_boxes
    dets = face_service.detect_and_recognize(
        frame,
        min_match_score=0.30,
        min_face_size=12,
        det_score_thresh=0.20,
        use_temporal_smoothing=False,
        enable_distant_mode=True,
        person_boxes=[person_box],
    )
    # The synthetic frame has no actual faces, so it shouldn't crash and should return a clean list
    assert isinstance(dets, list)
    print(f"[PASS] detect_and_recognize with person_boxes and distant_mode executed cleanly (found {len(dets)}).")


def test_faces_scan_distant_endpoint():
    """Test POST /api/faces/scan with enable_distant_mode=True."""
    client = TestClient(app)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)
    files = {"file": ("surveillance_frame.jpg", buf.tobytes(), "image/jpeg")}
    data = {
        "min_match_score": "0.35",
        "min_face_size": "14",
        "enable_distant_mode": "true",
    }
    resp = client.post("/api/faces/scan", files=files, data=data)
    assert resp.status_code == 200
    res = resp.json()
    assert res.get("success") is True
    assert "faces" in res
    print("[PASS] POST /api/faces/scan with enable_distant_mode returned 200 OK.")


def test_exclusive_one_to_one_resolution():
    """Verify that two people in the same frame cannot both be identified as the same person."""
    # Simulate two detected faces in the same frame both claiming "hariom"
    # Person 1 has higher match evidence (63%), Person 2 has lower (54%)
    det_person1 = {
        "name": "hariom",
        "is_known": True,
        "match_score": 0.40,
        "calibrated_conf": 63.0,
        "bbox": [200.0, 100.0, 240.0, 150.0],
    }
    det_person2 = {
        "name": "hariom",
        "is_known": True,
        "match_score": 0.35,
        "calibrated_conf": 54.0,
        "bbox": [400.0, 100.0, 440.0, 150.0],
    }

    # 1. Test direct enforcement method
    resolved = face_service._enforce_one_to_one_identities([det_person1, det_person2], min_match_score=0.32)
    assert len(resolved) == 2
    assert resolved[0]["name"] == "hariom" and resolved[0]["is_known"] is True
    assert resolved[1]["name"] == "Unknown" and resolved[1]["is_known"] is False

    # 2. Test through temporal smoothing (ensure smoothing never resurrects duplicate identity)
    face_service.tracks.clear()
    import time
    now = time.time()
    smoothed = face_service._apply_temporal_smoothing([det_person1, det_person2], now=now, min_match_score=0.32)
    known_names = [d["name"] for d in smoothed if d.get("is_known")]
    assert known_names == ["hariom"], f"Expected exactly one 'hariom', got {known_names}"
    print("[PASS] 1-to-1 identity conflict resolution prevents duplicate identity in same frame.")



def test_identity_stability_hysteresis():
    """
    Verify the complete identity stability contract:
    1. PRE-LOCK: even the very first Unknown frame after a confirmed frame is bridged
    2. POST-LOCK: Unknown frames within drop_tolerance are bridged  
    3. ALTERNATING: the real-world hariom/Unknown/hariom/Unknown pattern shows NO flicker
    4. RELEASE: lock releases cleanly after sustained Unknown frames
    """
    import time
    import copy

    face_service.tracks.clear()
    now = time.time()
    min_score = 0.32

    def make_det(name, is_known, score=0.42, calibrated=68.0):
        return {
            "name": name,
            "is_known": is_known,
            "match_score": score,
            "calibrated_conf": calibrated,
            "bbox": [200.0, 100.0, 250.0, 160.0],
            "is_threat": False,
            "is_ambiguous": False,
            "confidence": 0.9,
            "quality": 0.7,
            "l2_distance": 0.0,
            "type": "face",
            "class_name": f"Face: {name}" if is_known else "Face: Unknown",
            "threat_level": "NONE",
            "category": "Authorized Personnel" if is_known else "Unknown",
        }

    # --- TEST 1: PRE-LOCK bridging (hariom on frame 1, Unknown on frame 2, before lock is established) ---
    face_service.tracks.clear()
    results = []
    sequence = ["hariom", "Unknown", "hariom", "Unknown"]  # Alternating before lock
    for i, label in enumerate(sequence):
        is_known = label != "Unknown"
        det = make_det(label, is_known, score=0.42 if is_known else 0.1, calibrated=68.0 if is_known else 15.0)
        res = face_service._apply_temporal_smoothing([copy.deepcopy(det)], now=now + i * 0.1, min_match_score=min_score)
        results.append(res[0]["name"] if res else "NO_RESULT")

    print(f"  Alternating pattern results: {results}")
    assert all(n == "hariom" for n in results), \
        f"Alternating hariom/Unknown should always show 'hariom', got: {results}"
    print("[PASS] Pre-lock bridge: alternating hariom/Unknown shows no flicker.")

    # --- TEST 2: POST-LOCK bridging (multiple Unknown frames within tolerance) ---
    face_service.tracks.clear()
    results2 = []
    for i in range(3):  # 3 confirmed frames
        det = make_det("hariom", True)
        res = face_service._apply_temporal_smoothing([copy.deepcopy(det)], now=now + i * 0.1, min_match_score=min_score)
        results2.append(res[0]["name"] if res else "NO_RESULT")

    for i in range(3, 3 + face_service.identity_drop_tolerance):  # Unknown within tolerance
        det = make_det("Unknown", False, score=0.1, calibrated=15.0)
        res = face_service._apply_temporal_smoothing([copy.deepcopy(det)], now=now + i * 0.1, min_match_score=min_score)
        results2.append(res[0]["name"] if res else "NO_RESULT")

    print(f"  Post-lock bridge ({face_service.identity_drop_tolerance} Unknown frames): {results2}")
    # ALL frames within tolerance should show "hariom"
    within_tol = results2[3: 3 + face_service.identity_drop_tolerance]
    assert all(n == "hariom" for n in within_tol), \
        f"Lock should hold 'hariom' within drop_tolerance, got: {within_tol}"
    print(f"[PASS] Post-lock bridge: {face_service.identity_drop_tolerance} Unknown frames held as 'hariom'.")

    # --- TEST 3: LOCK RELEASE after sustained Unknown ---
    for i in range(3 + face_service.identity_drop_tolerance,
                   3 + face_service.identity_drop_tolerance + 5):  # Well beyond tolerance
        det = make_det("Unknown", False, score=0.1, calibrated=15.0)
        res = face_service._apply_temporal_smoothing([copy.deepcopy(det)], now=now + i * 0.1, min_match_score=min_score)
        results2.append(res[0]["name"] if res else "NO_RESULT")

    last_frames = results2[-3:]
    assert any(n == "Unknown" for n in last_frames), \
        f"After tolerance exceeded, lock should release to 'Unknown': {last_frames}"
    print("[PASS] Lock release: identity releases cleanly after sustained absence.")


if __name__ == "__main__":
    print("--- Running Distant Facial Recognition Test Suite ---")
    test_enhance_aligned_face_distant()
    test_deduplicate_face_detections()
    test_exclusive_one_to_one_resolution()
    test_identity_stability_hysteresis()
    test_person_guided_roi_and_distant_mode()
    test_faces_scan_distant_endpoint()
    print("ALL DISTANT FACIAL RECOGNITION TESTS PASSED!")


