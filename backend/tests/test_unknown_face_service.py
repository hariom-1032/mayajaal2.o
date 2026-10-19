import os
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
from fastapi.testclient import TestClient
import numpy as np

from app.main import app
from app.pipeline.unknown_face_service import unknown_face_service


def test_unknown_face_lifecycle():
    client = TestClient(app)

    # Clean slate for testing
    unknown_face_service.clear_all()

    # 1. Create a synthetic test image with a face-like area
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 50, (60, 60, 60), -1)

    # Generate a realistic 512-D L2-normalized embedding for Person 1
    rng = np.random.RandomState(42)
    person1_emb = rng.randn(512).astype(np.float32)
    person1_emb /= np.linalg.norm(person1_emb)

    face_frame1 = {
        "name": "Unknown",
        "is_known": False,
        "confidence": 0.942,
        "calibrated_conf": 18.5,  # Raw match score against suspects (should be ignored!)
        "bbox": [270.0, 190.0, 370.0, 290.0],
        "embedding": person1_emb,
    }

    # 2. Record unknown face on first camera (SITAICS)
    rec1 = unknown_face_service.record_unknown(
        frame=frame,
        face=face_frame1,
        camera_id="cam-sitaics",
        camera_name="SITAICS Gate 1",
    )
    assert rec1 is not None
    assert rec1["id"].startswith("UNK-")
    assert rec1["camera_id"] == "cam-sitaics"
    assert rec1["camera_name"] == "SITAICS Gate 1"
    assert rec1["seen_count"] == 1
    assert "SITAICS Gate 1" in rec1["cameras_seen"]
    # Check that confidence displays face detector confidence (~94.2%), NOT suspect calibrated conf (18.5%)
    assert rec1["confidence"] == 94.2
    assert "timestamp" in rec1
    assert rec1["photo_filename"]
    assert Path(unknown_face_service.photos_dir / rec1["photo_filename"]).exists()

    unk_id1 = rec1["id"]
    photo_fn1 = rec1["photo_filename"]
    print(f"[PASS] Successfully recorded initial unknown person {unk_id1} on SITAICS (Conf: {rec1['confidence']}%)")

    # 3. Simulate same person appearing on SECOND camera (SASET) with slight embedding perturbation (e.g. angle/lighting change)
    # Cosine similarity between emb1 and emb1_perturbed will be ~0.85 (well above 0.38 threshold)
    orth = rng.randn(512).astype(np.float32)
    orth -= np.dot(orth, person1_emb) * person1_emb
    orth /= np.linalg.norm(orth)
    person1_emb_perturbed = (0.65 * person1_emb + 0.35 * orth).astype(np.float32)
    person1_emb_perturbed /= np.linalg.norm(person1_emb_perturbed)

    face_frame2 = {
        "name": "Unknown",
        "is_known": False,
        "confidence": 0.965,
        "calibrated_conf": 24.1,
        "bbox": [280.0, 200.0, 380.0, 300.0],
        "embedding": person1_emb_perturbed,
    }

    rec2 = unknown_face_service.record_unknown(
        frame=frame,
        face=face_frame2,
        camera_id="cam-saset",
        camera_name="SASET Main Corridor",
    )
    assert rec2 is not None
    # Crucial assertion: MUST MATCH THE EXACT SAME UNKNOWN ID WITHOUT DUPLICATION!
    assert rec2["id"] == unk_id1, f"Expected {unk_id1} but got duplicate ID {rec2['id']}"
    assert rec2["seen_count"] == 2
    assert "SASET Main Corridor" in rec2["cameras_seen"]
    assert "SITAICS Gate 1" in rec2["cameras_seen"]
    assert rec2["confidence"] == 96.5  # Upgraded to higher detector confidence

    print(f"[PASS] Re-ID successfully matched same face on SASET! Total seen: {rec2['seen_count']}, cameras: {rec2['cameras_seen']}")

    # 4. Simulate a THIRD frame on SASET within rapid succession
    face_frame3 = {
        "name": "Unknown",
        "is_known": False,
        "confidence": 0.950,
        "calibrated_conf": 21.0,
        "bbox": [290.0, 205.0, 390.0, 305.0],
        "embedding": person1_emb,
    }
    rec3 = unknown_face_service.record_unknown(
        frame=frame,
        face=face_frame3,
        camera_id="cam-saset",
        camera_name="SASET Main Corridor",
    )
    assert rec3["id"] == unk_id1
    assert rec3["seen_count"] == 3
    print(f"[PASS] Rapid-fire frame successfully updated occurrence count to {rec3['seen_count']} without creating duplicates.")

    # 5. Now simulate a DIFFERENT person (Person 2) with an orthogonal embedding
    person2_emb = rng.randn(512).astype(np.float32)
    person2_emb /= np.linalg.norm(person2_emb)

    face_person2 = {
        "name": "Unknown",
        "is_known": False,
        "confidence": 0.920,
        "calibrated_conf": 12.0,
        "bbox": [150.0, 120.0, 250.0, 220.0],
        "embedding": person2_emb,
    }

    rec_p2 = unknown_face_service.record_unknown(
        frame=frame,
        face=face_person2,
        camera_id="cam-saset",
        camera_name="SASET Main Corridor",
    )
    assert rec_p2 is not None
    assert rec_p2["id"] != unk_id1, "Person 2 must receive a new distinct Unknown ID"
    unk_id2 = rec_p2["id"]
    print(f"[PASS] Distinct individual received new distinct ID {unk_id2}")

    # Check total count in database: must be exactly 2 individuals!
    assert len(unknown_face_service.records) == 2, f"Expected 2 unique unknown records, but got {len(unknown_face_service.records)}"
    print(f"[PASS] Total unique unknown persons in database: {len(unknown_face_service.records)} (0 duplicates!)")

    # 6. Test API GET /api/faces/unknown
    res = client.get("/api/faces/unknown")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["total"] == 2
    print("[PASS] GET /api/faces/unknown successfully retrieved recorded unknowns.")

    # 7. Test API GET /api/faces/unknown/{unknown_id}
    res_single = client.get(f"/api/faces/unknown/{unk_id1}")
    assert res_single.status_code == 200
    assert res_single.json()["unknown"]["id"] == unk_id1
    assert res_single.json()["unknown"]["seen_count"] == 3
    print(f"[PASS] GET /api/faces/unknown/{unk_id1} verified.")

    # 8. Test API GET /api/faces/unknown/photo/{filename}
    res_photo = client.get(f"/api/faces/unknown/photo/{photo_fn1}")
    assert res_photo.status_code == 200
    assert res_photo.headers["content-type"] == "image/jpeg"
    print(f"[PASS] GET /api/faces/unknown/photo/{photo_fn1} serves image correctly.")

    # 9. Test API GET /api/faces/unknown/stats
    res_stats = client.get("/api/faces/unknown/stats")
    assert res_stats.status_code == 200
    assert res_stats.json()["total_unknowns"] == 2
    print("[PASS] GET /api/faces/unknown/stats returned aggregated metrics.")

    # 10. Test API POST /api/faces/unknown/consolidate
    res_cons = client.post("/api/faces/unknown/consolidate?similarity_threshold=0.18")
    assert res_cons.status_code == 200
    assert res_cons.json()["total_remaining"] == 2
    print(f"[PASS] POST /api/faces/unknown/consolidate: {res_cons.json()['message']}")

    # 11. Test API PATCH /api/faces/unknown/{unknown_id}
    res_patch = client.patch(f"/api/faces/unknown/{unk_id1}", json={"status": "flagged", "notes": "Loitering near perimeter"})
    assert res_patch.status_code == 200
    assert res_patch.json()["unknown"]["status"] == "flagged"
    print("[PASS] PATCH /api/faces/unknown successfully updated status and notes.")

    # 12. Test API DELETE /api/faces/unknown/{unknown_id}
    res_del = client.delete(f"/api/faces/unknown/{unk_id1}")
    assert res_del.status_code == 200
    assert not Path(unknown_face_service.photos_dir / photo_fn1).exists()
    print("[PASS] DELETE /api/faces/unknown deleted record and photo file cleanly.")


if __name__ == "__main__":
    print("--- Running Unknown Face Service Re-ID & Multi-Camera Test Suite ---")
    test_unknown_face_lifecycle()
    print("\n[ALL PASS] ALL BIOMETRIC RE-ID & UNKNOWN SERVICE TESTS PASSED PERFECTLY!")
