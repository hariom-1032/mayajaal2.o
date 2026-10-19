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


def test_exclusive_one_to_one_identity_resolution():
    """Verify two people in the same frame can never both be assigned the same identity."""
    # Reset tracks to a clean state
    face_service.tracks.clear()
    face_service.next_track_id = 1

    # Simulate two people walking side-by-side both weakly matching "hariom"
    det1 = {
        "bbox": [100.0, 100.0, 140.0, 150.0],
        "confidence": 0.85,
        "match_score": 0.63,
        "calibrated_conf": 63.0,
        "name": "hariom",
        "is_known": True,
        "is_threat": False,
        "threat_level": "LOW",
        "category": "Authorized Personnel",
        "class_name": "Face: hariom",
    }
    det2 = {
        "bbox": [250.0, 100.0, 290.0, 150.0],
        "confidence": 0.80,
        "match_score": 0.54,
        "calibrated_conf": 54.0,
        "name": "hariom",
        "is_known": True,
        "is_threat": False,
        "threat_level": "LOW",
        "category": "Authorized Personnel",
        "class_name": "Face: hariom",
    }

    results = face_service._apply_temporal_smoothing([det1, det2], now=100.0, min_match_score=0.32)

    # Invariant: exactly one detection can have name "hariom"
    hariom_matches = [d for d in results if d["name"] == "hariom" and d["is_known"]]
    unknown_matches = [d for d in results if d["name"] == "Unknown"]

    assert len(hariom_matches) == 1, f"Expected 1 'hariom', got {len(hariom_matches)}"
    assert len(unknown_matches) == 1, f"Expected 1 'Unknown', got {len(unknown_matches)}"
    # Higher match score (0.63) must win the identity
    assert hariom_matches[0]["match_score"] == 0.63
    print("[PASS] 1-to-1 identity conflict resolution successfully prevented duplicate 'hariom' assignment.")


def test_identity_persistence_and_flicker_suppression():
    """Verify that when an identified person momentarily has a sub-threshold score (e.g. turned head),
    coasting/hysteresis holds the identity and suppresses flickering between name and Unknown."""
    face_service.tracks.clear()
    face_service.next_track_id = 1

    # Frame 1: Person is clearly identified
    det_f1 = {
        "bbox": [100.0, 100.0, 140.0, 150.0],
        "confidence": 0.85,
        "match_score": 0.38,
        "calibrated_conf": 65.0,
        "name": "hariom",
        "is_known": True,
        "is_threat": False,
        "threat_level": "LOW",
        "category": "Authorized Personnel",
        "class_name": "Face: hariom",
    }
    res1 = face_service._apply_temporal_smoothing([det_f1], now=1.0, min_match_score=0.32)
    assert res1[0]["name"] == "hariom"
    assert res1[0]["is_known"] is True

    # Frame 2 (0.3s later): Person turned head slightly, raw detection is "Unknown" (score 0.28)
    det_f2 = {
        "bbox": [105.0, 102.0, 145.0, 152.0],
        "confidence": 0.75,
        "match_score": 0.28,
        "calibrated_conf": 40.0,
        "name": "Unknown",
        "is_known": False,
        "is_threat": False,
        "threat_level": "NONE",
        "category": "Unknown",
        "class_name": "Face: Unknown",
    }
    res2 = face_service._apply_temporal_smoothing([det_f2], now=1.3, min_match_score=0.32)
    # Hysteresis should COAST and HOLD "hariom" instead of flickering to "Unknown"
    assert res2[0]["name"] == "hariom", f"Expected 'hariom' coasting, got {res2[0]['name']}"
    assert res2[0]["is_known"] is True
    print("[PASS] Identity coasting prevented flicker during momentary sub-threshold dip.")


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


if __name__ == "__main__":
    print("--- Running Distant Facial Recognition Test Suite ---")
    test_enhance_aligned_face_distant()
    test_deduplicate_face_detections()
    test_exclusive_one_to_one_identity_resolution()
    test_identity_persistence_and_flicker_suppression()
    test_person_guided_roi_and_distant_mode()
    test_faces_scan_distant_endpoint()
    print("ALL DISTANT FACIAL RECOGNITION TESTS PASSED!")
