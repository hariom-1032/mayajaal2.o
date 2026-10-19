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

    # 1. Create a synthetic test image with a face-like area
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (320, 240), 50, (60, 60, 60), -1)

    face = {
        "name": "Unknown",
        "is_known": False,
        "confidence": 0.91,
        "calibrated_conf": 88.0,
        "bbox": [270.0, 190.0, 370.0, 290.0],
        "track_id": 42,
    }

    # 2. Record unknown face
    rec = unknown_face_service.record_unknown(
        frame=frame,
        face=face,
        camera_id="cam-test-01",
        camera_name="Test North Gate",
    )
    assert rec is not None
    assert rec["id"].startswith("UNK-")
    assert rec["camera_id"] == "cam-test-01"
    assert rec["camera_name"] == "Test North Gate"
    assert "timestamp" in rec
    assert rec["photo_filename"]
    assert Path(unknown_face_service.photos_dir / rec["photo_filename"]).exists()

    unk_id = rec["id"]
    photo_fn = rec["photo_filename"]
    print(f"[PASS] Successfully recorded unknown person {unk_id} with photo {photo_fn}")

    # 3. Test API GET /api/faces/unknown
    res = client.get("/api/faces/unknown")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert any(u["id"] == unk_id for u in data["unknowns"])
    print("[PASS] GET /api/faces/unknown successfully retrieved recorded unknown.")

    # 4. Test API GET /api/faces/unknown/{unknown_id}
    res_single = client.get(f"/api/faces/unknown/{unk_id}")
    assert res_single.status_code == 200
    assert res_single.json()["unknown"]["id"] == unk_id
    print(f"[PASS] GET /api/faces/unknown/{unk_id} verified.")

    # 5. Test API GET /api/faces/unknown/photo/{filename}
    res_photo = client.get(f"/api/faces/unknown/photo/{photo_fn}")
    assert res_photo.status_code == 200
    assert res_photo.headers["content-type"] == "image/jpeg"
    print(f"[PASS] GET /api/faces/unknown/photo/{photo_fn} serves image correctly.")

    # 6. Test API GET /api/faces/unknown/stats
    res_stats = client.get("/api/faces/unknown/stats")
    assert res_stats.status_code == 200
    assert res_stats.json()["total_unknowns"] >= 1
    print("[PASS] GET /api/faces/unknown/stats returned aggregated metrics.")

    # 7. Test API PATCH /api/faces/unknown/{unknown_id}
    res_patch = client.patch(f"/api/faces/unknown/{unk_id}", json={"status": "flagged", "notes": "Suspicious behavior near gate"})
    assert res_patch.status_code == 200
    assert res_patch.json()["unknown"]["status"] == "flagged"
    print("[PASS] PATCH /api/faces/unknown successfully updated status and notes.")

    # 8. Test API DELETE /api/faces/unknown/{unknown_id}
    res_del = client.delete(f"/api/faces/unknown/{unk_id}")
    assert res_del.status_code == 200
    assert not Path(unknown_face_service.photos_dir / photo_fn).exists()
    print("[PASS] DELETE /api/faces/unknown deleted record and file cleanly.")


if __name__ == "__main__":
    print("--- Running Unknown Face Service Test Suite ---")
    test_unknown_face_lifecycle()
    print("ALL UNKNOWN FACE SERVICE TESTS PASSED!")
