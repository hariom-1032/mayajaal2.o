import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from app.pipeline.face_service import face_service


def test_facemesh_enrollment_pipeline():
    """Verify FaceMesh V2_478 gates face enrollment with sub-pixel landmarks and quarter profile support."""
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

    test_person_name = "__Gravitas_Mesh_SideView_Test__"

    # Register face (Primary sample)
    reg_result = face_service.register_face(
        image_bgr=img,
        name=test_person_name,
        is_suspect=False,
        threat_level="LOW",
        category="Authorized Personnel",
        notes="Automated Gravitas FaceMesh side-view verification",
    )

    print("Registration Result:", reg_result)
    assert reg_result.get("success") is True, f"Registration failed: {reg_result.get('error')}"
    assert reg_result.get("mesh_verified") is True, "Registration must be verified by FaceMesh dense 3D mesh"
    
    metrics = reg_result.get("mesh_metrics", {})
    assert "presence_score" in metrics, "Presence score must be in metrics"
    assert metrics["presence_score"] >= 0.70, f"Presence score too low: {metrics['presence_score']}"
    assert "ear_left" in metrics and "ear_right" in metrics, "EAR metrics must be present"
    assert metrics["ear_left"] >= 0.15 and metrics["ear_right"] >= 0.15, "Both eyes must be visible"
    assert "yaw_deg" in metrics and "pitch_deg" in metrics and "roll_deg" in metrics, "3D pose angles must be present"
    assert abs(metrics["yaw_deg"]) <= 35.0, f"Yaw exceeded quarter profile limit: {metrics['yaw_deg']}"
    assert abs(metrics["pitch_deg"]) <= 18.0, f"Pitch out of bounds: {metrics['pitch_deg']}"
    assert abs(metrics["roll_deg"]) <= 18.0, f"Roll out of bounds: {metrics['roll_deg']}"
    assert "mar" in metrics and metrics["mar"] <= 0.42, f"Mouth expression not neutral: {metrics['mar']}"
    assert "pose_type" in metrics and metrics["pose_type"] in ["FRONTAL", "QUARTER_PROFILE_LEFT", "QUARTER_PROFILE_RIGHT"], f"Invalid pose_type: {metrics.get('pose_type')}"

    print(f"[PASS] FaceMesh enrollment successfully verified with pose_type '{metrics['pose_type']}': {metrics}")

    # 3. Test adding an additional angle sample to the existing profile
    person_id = reg_result["person_id"]
    add_res = face_service.add_sample(person_id, img)
    assert add_res.get("success") is True, f"add_sample failed: {add_res.get('error')}"
    assert add_res.get("is_new") is False, "Sample added to existing person must have is_new=False"
    assert add_res.get("sample_count") > reg_result.get("sample_count"), "Sample count must increase"
    print(f"[PASS] Successfully added secondary angle sample to profile ({add_res['sample_count']} samples total)")

    # 4. Clean up test user
    del_result = face_service.delete_person(person_id)
    assert del_result.get("success") is True, "Failed to clean up test user"
    print(f"[PASS] Cleaned up temporary test person ID: {person_id}")


if __name__ == "__main__":
    print("Running FaceMesh Side-View & Enrollment Integration Tests...")
    test_facemesh_enrollment_pipeline()
    print("ALL FACEMESH ENROLLMENT TESTS PASSED!")
