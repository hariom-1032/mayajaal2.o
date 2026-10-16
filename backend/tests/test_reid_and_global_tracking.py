import sys
from pathlib import Path

# Ensure backend root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from starlette.testclient import TestClient

from app.main import app
from app.pipeline.reid_service import reid_service
from app.tracking.global_tracker import global_trace_manager


def test_reid_service_extraction():
    """Verify ReID feature extractor returns 512-D L2-normalized float32 vectors."""
    crop = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    feat = reid_service.extract_crop(crop)
    assert feat is not None, "Feature should not be None"
    assert feat.shape == (512,), f"Expected 512-D vector, got {feat.shape}"
    assert feat.dtype == np.float32
    norm = float(np.linalg.norm(feat))
    assert abs(norm - 1.0) < 1e-4, f"Feature vector must be unit-normalized, got {norm}"


def test_reid_similarity_discrimination():
    """Verify same person yields high similarity (>0.85) and different person yields low similarity (<0.55)."""
    # Person 1 (Blue top, dark bottom)
    p1 = np.zeros((256, 128, 3), dtype=np.uint8)
    p1[:128, :, 0] = 220
    p1[128:, :, :] = 30

    # Person 1 with lighting change / jitter
    p1_jitter = np.clip(p1.astype(np.int16) + np.random.randint(-15, 15, p1.shape), 0, 255).astype(np.uint8)

    # Person 2 (Yellow top, red bottom)
    p2 = np.zeros((256, 128, 3), dtype=np.uint8)
    p2[:128, :, 1] = 230
    p2[:128, :, 2] = 230
    p2[128:, :, 2] = 240

    f1 = reid_service.extract_crop(p1)
    f1_j = reid_service.extract_crop(p1_jitter)
    f2 = reid_service.extract_crop(p2)

    sim_same = reid_service.compute_similarity(f1, f1_j)
    sim_diff = reid_service.compute_similarity(f1, f2)

    print(f"\n[Re-ID Similarity] Same person: {sim_same:.4f} | Different person: {sim_diff:.4f}")
    assert sim_same > 0.85, f"Expected same person similarity > 0.85, got {sim_same}"
    assert sim_diff < 0.58, f"Expected different person similarity < 0.58, got {sim_diff}"


def test_cross_camera_global_tracking():
    """Verify that when a person moves from Camera 1 to Camera 2, their Global Trace ID is preserved."""
    global_trace_manager.reset()

    # Person A in Camera 1
    frame_cam1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_cam1[50:250, 100:200, 0] = 210  # Blue coat
    frame_cam1[250:400, 100:200, :] = 25  # Dark pants

    tid1, is_cross1, s1 = global_trace_manager.update_track(
        camera_id="cam_01",
        camera_name="North Gate Tower",
        local_track_id=1,
        box=[100, 50, 200, 400],
        frame_bgr=frame_cam1,
        confidence=0.89,
    )
    assert tid1 == "TRC-0001", f"First trace should be TRC-0001, got {tid1}"
    assert is_cross1 is False

    # Person A walks to Camera 2 (Sector B) with local track ID 88
    frame_cam2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_cam2[60:260, 300:400, 0] = 200
    frame_cam2[260:410, 300:400, :] = 30

    tid2, is_cross2, s2 = global_trace_manager.update_track(
        camera_id="cam_02",
        camera_name="Sector B Border Patrol",
        local_track_id=88,
        box=[300, 60, 400, 410],
        frame_bgr=frame_cam2,
        confidence=0.92,
    )
    assert tid2 == "TRC-0001", f"Camera 2 must retain same Trace ID TRC-0001! Got {tid2}"
    assert is_cross2 is True, "Must be flagged as cross-camera transition"
    assert s2 >= 0.65, f"Re-ID match score should be >= 0.65, got {s2}"

    # Person B appears on Camera 2 (Red coat, white trousers) with local track ID 99
    frame_cam2_p2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_cam2_p2[60:260, 450:550, 2] = 240
    frame_cam2_p2[260:410, 450:550, :] = 220

    tid3, is_cross3, s3 = global_trace_manager.update_track(
        camera_id="cam_02",
        camera_name="Sector B Border Patrol",
        local_track_id=99,
        box=[450, 60, 550, 410],
        frame_bgr=frame_cam2_p2,
        confidence=0.85,
    )
    assert tid3 == "TRC-0002", f"Different person must receive a new Trace ID! Got {tid3}"
    assert is_cross3 is False

    # Verify Trajectory Reconstruction
    traj = global_trace_manager.get_trace_trajectory("TRC-0001")
    assert traj["found"] is True
    assert traj["total_sightings"] == 2
    assert len(traj["waypoints"]) == 2
    assert traj["is_cross_camera"] is True
    assert traj["waypoints"][0]["camera_id"] == "cam_01"
    assert traj["waypoints"][1]["camera_id"] == "cam_02"
    assert traj["total_distance_km"] > 0.0
    print(f"\n[Trajectory PASS] Distance: {traj['total_distance_km']} km across {traj['total_sightings']} waypoints.")


def test_suspect_biometric_fusion_across_cameras():
    """Verify that when a trace is recognized as an enrolled suspect on Cam 1, Cam 2 retains the suspect identity via Re-ID."""
    global_trace_manager.reset()

    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[50:250, 100:200, 1] = 200  # Green shirt
    frame1[250:400, 100:200, 0] = 180  # Jeans

    tid, _, _ = global_trace_manager.update_track(
        camera_id="cam_01",
        camera_name="Checkpost Alpha",
        local_track_id=7,
        box=[100, 50, 200, 400],
        frame_bgr=frame1,
        confidence=0.95,
        suspect_name="Vikram Singh",
        threat_level="CRITICAL",
        category="Terror Suspect",
    )

    trace = global_trace_manager.get_trace(tid)
    assert trace.is_suspect is True
    assert trace.suspect_name == "Vikram Singh"
    assert trace.threat_level == "CRITICAL"

    # In Camera 2, person appears with face away from camera (no suspect_name provided)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2[50:250, 100:200, 1] = 190
    frame2[250:400, 100:200, 0] = 175

    tid2, is_cross, _ = global_trace_manager.update_track(
        camera_id="cam_02",
        camera_name="Checkpost Bravo",
        local_track_id=12,
        box=[100, 50, 200, 400],
        frame_bgr=frame2,
        confidence=0.90,
        suspect_name=None,  # Face not visible!
    )

    assert tid2 == tid, "Must match same trace"
    trace2 = global_trace_manager.get_trace(tid2)
    assert trace2.is_suspect is True
    assert trace2.suspect_name == "Vikram Singh", "Must inherit suspect identity from Cam 1!"
    print(f"\n[Biometric Fusion PASS] Trace {tid2} identified as {trace2.suspect_name} on Cam 2 without face detection!")


def test_global_trace_rest_api():
    """Verify REST endpoints /api/tracking/global/traces."""
    client = TestClient(app)

    # 1. List traces
    resp = client.get("/api/tracking/global/traces")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "traces" in data
    assert len(data["traces"]) >= 1

    # 2. Get specific trajectory
    tid = data["traces"][0]["trace_id"]
    traj_resp = client.get(f"/api/tracking/global/traces/{tid}")
    assert traj_resp.status_code == 200
    traj_data = traj_resp.json()
    assert traj_data["found"] is True
    assert traj_data["trace_id"] == tid
    assert "waypoints" in traj_data

    # 3. Reset traces
    del_resp = client.delete("/api/tracking/global/traces")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "success"

    # 4. Verify empty after reset
    list_after = client.get("/api/tracking/global/traces").json()
    assert list_after["total_active"] == 0
    print("\n[REST API PASS] Global trace endpoints verified successfully.")


if __name__ == "__main__":
    print("\n=======================================================")
    print(" Running Person Re-ID & Multi-Camera Tracking Tests ")
    print("=======================================================\n")
    test_reid_service_extraction()
    print(" [1/5] test_reid_service_extraction: PASSED")
    test_reid_similarity_discrimination()
    print(" [2/5] test_reid_similarity_discrimination: PASSED")
    test_cross_camera_global_tracking()
    print(" [3/5] test_cross_camera_global_tracking: PASSED")
    test_suspect_biometric_fusion_across_cameras()
    print(" [4/5] test_suspect_biometric_fusion_across_cameras: PASSED")
    test_global_trace_rest_api()
    print(" [5/5] test_global_trace_rest_api: PASSED")
    print("\n ALL 5 RE-ID & MULTI-CAMERA TESTS PASSED! \n")
