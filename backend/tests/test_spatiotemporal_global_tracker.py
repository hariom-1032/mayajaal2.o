import sys
import time
from pathlib import Path

# Ensure backend root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from app.tracking.global_tracker import GlobalTraceManager, calc_haversine_km
from app.pipeline.reid_service import reid_service


def test_haversine_distance_computation():
    """Verify Great Circle haversine distance calculation."""
    # Jammu Border Post 1 to Post 2 (~1.1 km apart)
    lat1, lon1 = 32.7240, 74.6710
    lat2, lon2 = 32.7340, 74.6710
    dist_km = calc_haversine_km(lat1, lon1, lat2, lon2)
    print(f"\n[Haversine Test] Distance between test points: {dist_km:.3f} km")
    assert 1.0 < dist_km < 1.2, f"Expected ~1.11 km, got {dist_km}"


def test_feasible_pedestrian_transition():
    """Verify that a feasible walking speed (~5 km/h) between cameras allows trace association."""
    gtm = GlobalTraceManager(
        similarity_threshold=0.65,
        max_pedestrian_speed_kmh=25.0,
    )
    # Register two cameras 60 meters apart
    cam1_lat, cam1_lon = 32.72400, 74.67100
    # ~60m north
    cam2_lat, cam2_lon = 32.72454, 74.67100
    gtm.register_camera_coordinates("gate_01", cam1_lat, cam1_lon)
    gtm.register_camera_coordinates("pathway_02", cam2_lat, cam2_lon)

    # Identical synthetic person crop (Blue top, dark pants)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:250, 100:200, 0] = 220
    frame[250:400, 100:200, :] = 30

    t0 = 1700000000.0
    # Step 1: Seen on gate_01
    tid1, is_cross1, _ = gtm.update_track(
        camera_id="gate_01",
        camera_name="Gate 01",
        local_track_id=10,
        box=[100, 50, 200, 400],
        frame_bgr=frame,
        confidence=0.92,
        timestamp=t0,
    )
    assert tid1 == "TRC-0001"
    assert is_cross1 is False

    # Step 2: 45 seconds later on pathway_02 (speed = 0.06 km / (45/3600 h) = 4.8 km/h walking)
    t1 = t0 + 45.0
    tid2, is_cross2, score2 = gtm.update_track(
        camera_id="pathway_02",
        camera_name="Pathway 02",
        local_track_id=42,
        box=[120, 60, 220, 410],
        frame_bgr=frame,
        confidence=0.90,
        timestamp=t1,
    )

    assert tid2 == "TRC-0001", f"Feasible transit must match TRC-0001, got {tid2}"
    assert is_cross2 is True
    assert score2 >= 0.65

    # Check waypoint recorded the speed
    trace = gtm.get_trace("TRC-0001")
    assert trace is not None
    assert len(trace.waypoints) == 2
    wp2 = trace.waypoints[1]
    assert wp2["speed_kmh"] is not None
    assert 3.0 < wp2["speed_kmh"] < 8.0, f"Expected walking speed ~4.8 km/h, got {wp2['speed_kmh']}"
    print(f"\n[PASS] Feasible pedestrian transit accepted at {wp2['speed_kmh']:.1f} km/h.")


def test_teleportation_rejection_at_distance():
    """
    CRITICAL TEST: Verify that when two cameras are 2 km apart,
    an appearance match appearing only 10s later is REJECTED due to kinematic impossibility.
    """
    gtm = GlobalTraceManager(
        similarity_threshold=0.65,
        max_pedestrian_speed_kmh=25.0,
    )
    # Camera Post A
    gtm.register_camera_coordinates("post_alpha", 32.7240, 74.6710)
    # Camera Post B (~2.2 km away)
    gtm.register_camera_coordinates("post_bravo", 32.7440, 74.6710)

    # High-similarity person crop
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:250, 100:200, 1] = 230
    frame[250:400, 100:200, 2] = 210

    t0 = 1700000000.0
    tid1, _, _ = gtm.update_track(
        camera_id="post_alpha",
        camera_name="Post Alpha",
        local_track_id=1,
        box=[100, 50, 200, 400],
        frame_bgr=frame,
        confidence=0.95,
        timestamp=t0,
    )
    assert tid1 == "TRC-0001"

    # Person with IDENTICAL appearance appears at Post Bravo (2.2 km away) just 10 seconds later!
    # Required speed = 2.2 km / (10/3600) = 792 km/h (Physically impossible for human)
    t1 = t0 + 10.0
    tid2, is_cross2, _ = gtm.update_track(
        camera_id="post_bravo",
        camera_name="Post Bravo",
        local_track_id=99,
        box=[100, 50, 200, 400],
        frame_bgr=frame,
        confidence=0.95,
        timestamp=t1,
        is_vehicle=False,  # Pedestrian
    )

    # MUST BE REJECTED! Must initialize a brand new trace TRC-0002 rather than corrupting TRC-0001!
    assert tid2 == "TRC-0002", f"Teleportation candidate should be rejected and assigned TRC-0002, got {tid2}"
    assert is_cross2 is False
    print(f"\n[PASS] Teleportation rejected! TRC-0001 and TRC-0002 kept strictly isolated.")


def test_vehicle_mode_speed_distinction():
    """Verify that vehicle mode allows higher travel speeds (e.g. 70 km/h) that pedestrian mode rejects."""
    gtm = GlobalTraceManager(
        similarity_threshold=0.65,
        max_pedestrian_speed_kmh=25.0,
        max_vehicle_speed_kmh=100.0,
    )
    # 1.2 km road stretch
    gtm.register_camera_coordinates("highway_entry", 32.7240, 74.6710)
    gtm.register_camera_coordinates("highway_exit", 32.7350, 74.6710)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:250, 100:200, :] = 180

    t0 = 1700000000.0
    gtm.update_track(
        camera_id="highway_entry",
        camera_name="Highway Entry",
        local_track_id=5,
        box=[100, 50, 200, 400],
        frame_bgr=frame,
        confidence=0.90,
        timestamp=t0,
    )

    # 1.2 km in 65 seconds = ~66 km/h
    t1 = t0 + 65.0

    # 1. Test as pedestrian -> should be REJECTED (exceeds 25 km/h)
    gtm_ped = GlobalTraceManager(similarity_threshold=0.65, max_pedestrian_speed_kmh=25.0)
    gtm_ped.register_camera_coordinates("highway_entry", 32.7240, 74.6710)
    gtm_ped.register_camera_coordinates("highway_exit", 32.7350, 74.6710)
    gtm_ped.update_track("highway_entry", "Highway Entry", 5, [100, 50, 200, 400], frame, 0.9, timestamp=t0)
    tid_ped, is_cross_ped, _ = gtm_ped.update_track(
        "highway_exit", "Highway Exit", 7, [100, 50, 200, 400], frame, 0.9, timestamp=t1, is_vehicle=False
    )
    assert tid_ped == "TRC-0002", "Pedestrian limit should reject 66 km/h speed"
    assert is_cross_ped is False

    # 2. Test as vehicle -> should be ACCEPTED (under 100 km/h)
    tid_veh, is_cross_veh, _ = gtm.update_track(
        "highway_exit", "Highway Exit", 7, [100, 50, 200, 400], frame, 0.9, timestamp=t1, is_vehicle=True
    )
    assert tid_veh == "TRC-0001", "Vehicle limit should accept 66 km/h speed"
    assert is_cross_veh is True
    print(f"\n[PASS] Vehicle speed mode distinction verified successfully.")


def test_collocated_cameras_allow_fast_handoff():
    """Verify that cameras mounted on the same mast (<15m apart) allow immediate handoff."""
    gtm = GlobalTraceManager(
        similarity_threshold=0.65,
        spatial_collocation_threshold_m=15.0,
    )
    # Two cameras on the same mast (~3m apart)
    gtm.register_camera_coordinates("mast_north", 32.724000, 74.671000)
    gtm.register_camera_coordinates("mast_east", 32.724025, 74.671000)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:250, 100:200, 0] = 200
    frame[50:250, 100:200, 1] = 100

    t0 = 1700000000.0
    gtm.update_track("mast_north", "Mast North", 1, [100, 50, 200, 400], frame, 0.9, timestamp=t0)

    # Transition just 0.8s later (person turning around the corner of the mast)
    t1 = t0 + 0.8
    tid, is_cross, _ = gtm.update_track(
        "mast_east", "Mast East", 2, [100, 50, 200, 400], frame, 0.9, timestamp=t1
    )
    assert tid == "TRC-0001", f"Collocated cameras must permit rapid handoff, got {tid}"
    assert is_cross is True
    print(f"\n[PASS] Collocated camera handoff permitted without false speed alarms.")


if __name__ == "__main__":
    print("\n=======================================================")
    print(" Running Spatio-Temporal & Kinematic Gating Tests ")
    print("=======================================================\n")
    test_haversine_distance_computation()
    print(" [1/4] test_haversine_distance_computation: PASSED")
    test_feasible_pedestrian_transition()
    print(" [2/4] test_feasible_pedestrian_transition: PASSED")
    test_teleportation_rejection_at_distance()
    print(" [3/4] test_teleportation_rejection_at_distance: PASSED")
    test_vehicle_mode_speed_distinction()
    print(" [4/5] test_vehicle_mode_speed_distinction: PASSED")
    test_collocated_cameras_allow_fast_handoff()
    print(" [5/5] test_collocated_cameras_allow_fast_handoff: PASSED")
    print("\n ALL SPATIO-TEMPORAL & KINEMATIC TESTS PASSED! \n")
