import hashlib
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.face_service import face_service
from app.core.config import settings



def test_face_service_initialization():
    assert face_service.is_ready, "Face engine must be initialized and ready"
    if face_service.use_uniface:
        assert face_service.analyzer is not None, "UniFace analyzer must be initialized"
    else:
        assert face_service.detector is not None, "Face detector must be initialized"
        assert face_service.recognizer is not None, "Face recognizer must be initialized"
    
    persons = face_service.list_persons()
    assert len(persons) >= 1, "Should have at least 1 enrolled person"
    print(f"[PASS] FaceService initialized successfully with {len(persons)} enrolled person(s): {[p['name'] for p in persons]}.")


def test_face_api_endpoints():
    """Test the REST API endpoints using FastAPI TestClient."""
    client = TestClient(app)
    
    # 1. GET /api/faces
    resp = client.get("/api/faces")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("success") is True
    assert data.get("count") >= 1
    print("[PASS] GET /api/faces returned 200 OK with enrolled persons.")

    # 2. GET /api/faces/status
    resp = client.get("/api/faces/status")
    assert resp.status_code == 200
    status_data = resp.json()
    assert status_data.get("status") == "ready"
    assert status_data.get("detector_loaded") is True
    assert status_data.get("recognizer_loaded") is True
    print("[PASS] GET /api/faces/status returned 200 OK (status=ready).")

    # 3. GET /api/faces/thumbnail/{filename}
    thumb_path = face_service.thumbnails_dir / "person_1788701662_e0faf3.jpg"
    if thumb_path.exists():
        resp = client.get("/api/faces/thumbnail/person_1788701662_e0faf3.jpg")
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "image/jpeg"
        print("[PASS] GET /api/faces/thumbnail serving image/jpeg successfully.")

    # 4. POST /api/faces/scan with synthetic frame
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", dummy_img)
    files = {"file": ("test.jpg", buf.tobytes(), "image/jpeg")}
    resp = client.post("/api/faces/scan", files=files)
    assert resp.status_code == 200
    scan_data = resp.json()
    assert scan_data.get("success") is True
    assert "faces" in scan_data
    print("[PASS] POST /api/faces/scan executed successfully.")


def test_suspect_classification():
    """Test suspect classification, threat levels, and PATCH metadata updates."""
    client = TestClient(app)
    resp = client.get("/api/faces")
    assert resp.status_code == 200
    data = resp.json()
    persons = data.get("persons", [])
    assert len(persons) >= 1
    p = persons[0]
    pid = p["id"]

    # Verify suspect fields exist on enrolled person
    assert "is_suspect" in p
    assert "threat_level" in p
    assert "category" in p

    # Test PATCH metadata update
    patch_resp = client.patch(
        f"/api/faces/{pid}",
        json={
            "is_suspect": True,
            "threat_level": "CRITICAL",
            "category": "Wanted / BOLO",
            "notes": "Test BOLO incident"
        },
    )
    assert patch_resp.status_code == 200
    patch_data = patch_resp.json()
    assert patch_data.get("success") is True
    updated_p = patch_data["person"]
    assert updated_p["is_suspect"] is True
    assert updated_p["threat_level"] == "CRITICAL"
    assert updated_p["category"] == "Wanted / BOLO"
    assert updated_p["notes"] == "Test BOLO incident"

    # Reset back to non-suspect for test safety
    reset_resp = client.patch(
        f"/api/faces/{pid}",
        json={
            "is_suspect": False,
            "threat_level": "LOW",
            "category": "Authorized Personnel",
            "notes": ""
        },
    )
    assert reset_resp.status_code == 200
    assert reset_resp.json()["person"]["is_suspect"] is False
    print("[PASS] Suspect classification and PATCH metadata update verified successfully.")


if __name__ == "__main__":
    print("Running Facial Recognition Integration Tests...")
    test_face_service_initialization()
    test_face_api_endpoints()
    test_suspect_classification()
    print("ALL TESTS PASSED!")
