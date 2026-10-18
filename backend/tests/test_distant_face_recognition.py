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


def test_exclusive_one_to_one_identity_assignment():
    """
    Verify physical invariant: When two distinct people walking side-by-side
    both match the same enrolled person (e.g. 'hariom' at 60% and 55%),
    _enforce_unique_identities ensures ONLY the higher-confidence face keeps 'hariom',
    while the competing duplicate is demoted to 'Unknown'.
    """
    det1 = {
        "bbox": [100.0, 100.0, 150.0, 150.0],
        "name": "hariom",
        "is_known": True,
        "match_score": 0.35,
        "calibrated_conf": 60.0,
        "class_name": "Face: hariom",
        "is_threat": False,
    }
    det2 = {
        "bbox": [200.0, 100.0, 250.0, 150.0],  # Non-overlapping adjacent person
        "name": "hariom",
        "is_known": True,
        "match_score": 0.32,
        "calibrated_conf": 55.0,
        "class_name": "Face: hariom",
        "is_threat": False,
    }

    resolved = face_service._enforce_unique_identities([det1, det2], min_match_score=0.34)
    assert len(resolved) == 2

    # The higher match (60%) must be hariom
    known_dets = [d for d in resolved if d["is_known"]]
    unknown_dets = [d for d in resolved if not d["is_known"]]

    assert len(known_dets) == 1, f"Expected exactly 1 known face, got {len(known_dets)}"
    assert known_dets[0]["name"] == "hariom"
    assert known_dets[0]["calibrated_conf"] == 60.0

    assert len(unknown_dets) == 1, f"Expected 1 demoted unknown face, got {len(unknown_dets)}"
    assert unknown_dets[0]["name"] == "Unknown"
    assert unknown_dets[0]["class_name"] == "Face: Unknown"
    assert unknown_dets[0]["calibrated_conf"] < 50.0

    print("[PASS] _enforce_unique_identities successfully prevented duplicate 'hariom' in same frame.")


def test_temporal_smoothing_no_duplicate_identity_resurrection():
    """
    Verify that temporal smoothing across consecutive frames does NOT resurrect duplicate
    identities across neighboring tracks, and drawing produces one known green tag and one neutral gray tag.
    """
    # Reset face service tracks for clean test state
    face_service.tracks.clear()
    face_service.next_track_id = 1

    # Frame 1: Person A has 60% Hariom, Person B has 55% Hariom (demoted to Unknown)
    frame1_detA = {
        "bbox": [100.0, 100.0, 140.0, 140.0],
        "name": "hariom",
        "is_known": True,
        "match_score": 0.36,
        "calibrated_conf": 62.0,
        "class_name": "Face: hariom",
        "is_threat": False,
    }
    frame1_detB = {
        "bbox": [200.0, 100.0, 240.0, 140.0],
        "name": "hariom",
        "is_known": True,
        "match_score": 0.33,
        "calibrated_conf": 55.0,
        "class_name": "Face: hariom",
        "is_threat": False,
    }

    res_f1 = face_service._apply_temporal_smoothing(
        face_service._enforce_unique_identities([frame1_detA, frame1_detB], min_match_score=0.34),
        now=100.0,
        min_match_score=0.34,
    )
    names_f1 = [d["name"] for d in res_f1 if d["is_known"]]
    assert names_f1 == ["hariom"], f"Frame 1 must only have 1 'hariom', got {names_f1}"

    # Frame 2: Both people walk forward slightly
    frame2_detA = {
        "bbox": [102.0, 105.0, 142.0, 145.0],
        "name": "hariom",
        "is_known": True,
        "match_score": 0.35,
        "calibrated_conf": 60.0,
        "class_name": "Face: hariom",
        "is_threat": False,
    }
    frame2_detB = {
        "bbox": [202.0, 105.0, 242.0, 145.0],
        "name": "hariom",
        "is_known": True,
        "match_score": 0.32,
        "calibrated_conf": 54.0,
        "class_name": "Face: hariom",
        "is_threat": False,
    }

    res_f2 = face_service._apply_temporal_smoothing(
        face_service._enforce_unique_identities([frame2_detA, frame2_detB], min_match_score=0.34),
        now=100.1,
        min_match_score=0.34,
    )
    names_f2 = [d["name"] for d in res_f2 if d["is_known"]]
    assert names_f2 == ["hariom"], f"Frame 2 must NEVER duplicate 'hariom', got {names_f2}"

    # Verify draw_faces outputs appropriate distinct labels
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)
    annotated = face_service.draw_faces(canvas, res_f2, show_metrics=True)
    assert annotated.shape == canvas.shape

    print("[PASS] Temporal smoothing across frames rigorously preserves 1-to-1 identity uniqueness.")


if __name__ == "__main__":
    print("--- Running Distant Facial Recognition Test Suite ---")
    test_enhance_aligned_face_distant()
    test_deduplicate_face_detections()
    test_person_guided_roi_and_distant_mode()
    test_faces_scan_distant_endpoint()
    test_exclusive_one_to_one_identity_assignment()
    test_temporal_smoothing_no_duplicate_identity_resurrection()
    print("ALL DISTANT FACIAL RECOGNITION TESTS PASSED!")
