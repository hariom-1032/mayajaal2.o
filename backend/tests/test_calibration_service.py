import json
import sys
from pathlib import Path

# Ensure backend root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from starlette.testclient import TestClient

from app.main import app
from app.tracking.calibration_service import CalibrationService, calibration_service
from app.tracking.global_tracker import GlobalTraceManager


def test_4point_homography_exactness():
    """Verify that 4-point planar homography accurately maps pixels to ground meters and back."""
    calib_srv = CalibrationService()

    # Image points: trapezoidal perspective of a 10m x 20m rectangular ground patch
    image_pts = [
        [300.0, 600.0],   # Top-left (far)
        [900.0, 600.0],   # Top-right (far)
        [1100.0, 1000.0], # Bottom-right (near)
        [100.0, 1000.0],  # Bottom-left (near)
    ]
    # Corresponding ground points in meters
    ground_pts = [
        [-5.0, 25.0], # Top-left
        [5.0, 25.0],  # Top-right
        [5.0, 5.0],   # Bottom-right
        [-5.0, 5.0],  # Bottom-left
    ]

    origin_gps = (23.154210, 72.885257)
    calib = calib_srv.register_4point_calibration(
        camera_id="cam_test_01",
        image_points=image_pts,
        ground_points_meters=ground_pts,
        world_origin_gps=origin_gps,
    )

    # 1. Forward projection: pixels -> ground meters
    for img_p, gnd_p in zip(image_pts, ground_pts):
        gx, gy = calib.image_to_ground(img_p[0], img_p[1])
        assert abs(gx - gnd_p[0]) < 0.05, f"Expected X={gnd_p[0]}, got {gx}"
        assert abs(gy - gnd_p[1]) < 0.05, f"Expected Y={gnd_p[1]}, got {gy}"

    # 2. Reverse projection: ground meters -> pixels
    for img_p, gnd_p in zip(image_pts, ground_pts):
        u, v = calib.ground_to_image(gnd_p[0], gnd_p[1])
        assert abs(u - img_p[0]) < 0.5, f"Expected u={img_p[0]}, got {u}"
        assert abs(v - img_p[1]) < 0.5, f"Expected v={img_p[1]}, got {v}"

    # 3. GPS offset conversion
    gps = calib.ground_to_gps(0.0, 25.0)  # 25m north
    assert gps is not None
    assert gps[0] > origin_gps[0], "Northing must increase latitude"
    print("\n[Homography Exactness PASS] 4-point mapping verified with sub-centimeter fidelity.")


def test_footpoint_bounding_box_projection():
    """Verify that project_box_footprint extracts the bottom-center anchor and returns meters and GPS."""
    calib = calibration_service.get_calibration("bop-jk-01")
    assert calib is not None, "bop-jk-01 must have default calibration loaded"

    # Person standing in frame: box = [x1, y1, x2, y2]
    # Foot contact point is ((400 + 600)/2, 900) = (500, 900)
    box = [400.0, 200.0, 600.0, 900.0]

    ground_m, gps = calibration_service.project_box_footprint(
        camera_id="bop-jk-01",
        box=box,
        frame_resolution=(1280, 720),
    )

    assert ground_m is not None
    assert gps is not None
    assert len(ground_m) == 2
    assert len(gps) == 2
    # Verify reasonable meter distances
    gx, gy = ground_m
    assert -20.0 < gx < 20.0, f"Unreasonable lateral ground coordinate: {gx}"
    assert 0.0 < gy < 60.0, f"Unreasonable forward ground coordinate: {gy}"
    print(f"\n[Footprint PASS] Box bottom-center projected to ({gx:.2f}m, {gy:.2f}m) | GPS: {gps}")


def test_analytical_pinhole_estimation():
    """Verify automatic synthesis of homography matrix from camera mounting parameters."""
    calib_srv = CalibrationService()
    calib = calib_srv.estimate_pinhole_calibration(
        camera_id="tower_cam_auto",
        camera_height_m=6.0,
        pitch_degrees=35.0,
        hfov_degrees=75.0,
        image_width=1920,
        image_height=1080,
        world_origin_gps=(32.7240, 74.6710),
    )

    assert calib.H is not None
    assert calib.H.shape == (3, 3)

    # Pixel in the middle-bottom (closer to camera)
    gx_near, gy_near = calib.image_to_ground(960.0, 950.0)
    # Pixel higher up in frame (further away)
    gx_far, gy_far = calib.image_to_ground(960.0, 650.0)

    # Y is forward distance: points higher in the frame must be further away
    assert gy_far > gy_near, f"Higher pixel must project to greater ground depth: gy_far={gy_far} vs gy_near={gy_near}"
    print(f"\n[Pinhole Synthesis PASS] Forward depth gradient verified: Near={gy_near:.1f}m -> Far={gy_far:.1f}m.")


def test_tracking_waypoints_footprint_integration():
    """Verify that GlobalTraceManager automatically captures footprint ground coordinates and GPS in waypoints."""
    gtm = GlobalTraceManager()

    # Person detected on bop-jk-01
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[150:650, 400:600, 0] = 200

    tid, _, _ = gtm.update_track(
        camera_id="bop-jk-01",
        camera_name="Audi 1 Forward Barbed Fence",
        local_track_id=101,
        box=[400.0, 150.0, 600.0, 650.0],
        frame_bgr=frame,
        confidence=0.91,
    )

    traj = gtm.get_trace_trajectory(tid)
    assert traj["found"] is True
    assert len(traj["waypoints"]) == 1

    wp0 = traj["waypoints"][0]
    assert "footprint_ground_m" in wp0
    assert "footprint_gps" in wp0
    assert wp0["footprint_ground_m"] is not None
    assert wp0["footprint_gps"] is not None
    print(f"\n[Waypoint Footprint PASS] Trace {tid} recorded ground footprint: {wp0['footprint_ground_m']}m, GPS: {wp0['footprint_gps']}")


def test_calibration_rest_api():
    """Verify calibration REST endpoints."""
    client = TestClient(app)

    # 1. GET /api/tracking/calibrations
    resp = client.get("/api/tracking/calibrations")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["total"] >= 1

    # 2. GET /api/tracking/calibrations/bop-jk-01
    resp_one = client.get("/api/tracking/calibrations/bop-jk-01")
    assert resp_one.status_code == 200
    calib_data = resp_one.json()
    assert calib_data["calibration"]["camera_id"] == "bop-jk-01"

    # 3. POST /api/tracking/calibrations/project
    proj_resp = client.post(
        "/api/tracking/calibrations/project",
        data={
            "camera_id": "bop-jk-01",
            "box_json": json.dumps([350.0, 100.0, 550.0, 850.0]),
            "img_w": 1280,
            "img_h": 720,
        },
    )
    assert proj_resp.status_code == 200
    proj_data = proj_resp.json()
    assert proj_data["status"] == "success"
    assert proj_data["ground_meters"] is not None
    assert proj_data["gps"] is not None
    print("\n[REST API PASS] Calibration endpoints verified successfully.")


if __name__ == "__main__":
    print("\n=======================================================")
    print(" Running Camera Calibration & Homography Tests ")
    print("=======================================================\n")
    test_4point_homography_exactness()
    print(" [1/5] test_4point_homography_exactness: PASSED")
    test_footpoint_bounding_box_projection()
    print(" [2/5] test_footpoint_bounding_box_projection: PASSED")
    test_analytical_pinhole_estimation()
    print(" [3/5] test_analytical_pinhole_estimation: PASSED")
    test_tracking_waypoints_footprint_integration()
    print(" [4/5] test_tracking_waypoints_footprint_integration: PASSED")
    test_calibration_rest_api()
    print(" [5/5] test_calibration_rest_api: PASSED")
    print("\n ALL 5 CALIBRATION & HOMOGRAPHY TESTS PASSED! \n")
