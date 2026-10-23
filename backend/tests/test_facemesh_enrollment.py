import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from app.pipeline.face_service import face_service


def test_facemesh_enrollment_pipeline():
    """Verify FaceMesh V2_478 is active and gates face enrollment with sub-pixel landmarks."""
    # 1. Engine check
    assert face_service.is_ready, "FaceService must be operational"
    assert face_service.mesher is not None, "FaceMesh V2_478 must be initialized"
    assert "FaceMesh V2_478" in face_service.engine_name, "FaceMesh must be registered in engine_name"
    print(f"[PASS] FaceService active engine: {face_service.engine_name}")

    # 2. Test registration on an actual face thumbnail
    test_thumb = face_service.thumbnails_dir / "person_1789038743_adf5f2.jpg"
    assert test_thumb.exists(), "Test thumbnail must exist for enrollment verification"
    img = cv2.imread(str(test_thumb))
    assert img is not None and img.size > 0, "Image must be loaded successfully"

    test_person_name = "__Gravitas_Mesh_Test_User__"

    # Register face
    reg_result = face_service.register_face(
        image_bgr=img,
        name=test_person_name,
        is_suspect=False,
        threat_level="LOW",
        category="Authorized Personnel",
        notes="Automated Gravitas FaceMesh verification enrollment",
    )

    print("Registration Result:", reg_result)
    assert reg_result.get("success") is True, f"Registration failed: {reg_result.get('error')}"
    assert reg_result.get("mesh_verified") is True, "Registration must be verified by FaceMesh dense 3D mesh"
    
    metrics = reg_result.get("mesh_metrics", {})
    assert "presence_score" in metrics, "Presence score must be in metrics"
    assert metrics["presence_score"] >= 0.70, f"Presence score too low: {metrics['presence_score']}"
    assert "ear_left" in metrics and "ear_right" in metrics, "EAR metrics must be present"
    assert metrics["ear_left"] >= 0.18 and metrics["ear_right"] >= 0.18, "Eyes must be open"
    assert "yaw_deg" in metrics and "pitch_deg" in metrics and "roll_deg" in metrics, "3D pose angles must be present"
    assert abs(metrics["yaw_deg"]) <= 16.0, f"Yaw out of bounds: {metrics['yaw_deg']}"
    assert abs(metrics["pitch_deg"]) <= 16.0, f"Pitch out of bounds: {metrics['pitch_deg']}"
    assert abs(metrics["roll_deg"]) <= 16.0, f"Roll out of bounds: {metrics['roll_deg']}"
    assert "mar" in metrics and metrics["mar"] <= 0.42, f"Mouth expression not neutral: {metrics['mar']}"

    print(f"[PASS] FaceMesh enrollment successfully verified with metrics: {metrics}")

    # 3. Clean up test user
    person_id = reg_result["person_id"]
    del_result = face_service.delete_person(person_id)
    assert del_result.get("success") is True, "Failed to clean up test user"
    print(f"[PASS] Cleaned up temporary test person ID: {person_id}")


if __name__ == "__main__":
    print("Running FaceMesh Enrollment Integration Tests...")
    test_facemesh_enrollment_pipeline()
    print("ALL FACEMESH ENROLLMENT TESTS PASSED!")
