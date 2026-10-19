import sys
from pathlib import Path

# Ensure backend root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from starlette.testclient import TestClient

from app.main import app
from app.tracking.topology import CameraTopology, camera_topology
from app.tracking.global_tracker import GlobalTraceManager


def test_camera_topology_graph_initialization():
    """Verify camera topology graph loads default corridors and handles bidirectional lookups."""
    top = CameraTopology()
    assert top.is_adjacent("bop-jk-01", "bop-jk-02") is True
    assert top.is_adjacent("bop-jk-02", "bop-jk-01") is True  # Bidirectional
    assert top.is_adjacent("bop-jk-01", "bop-jk-05") is False  # Non-adjacent

    corridor = top.get_corridor("bop-jk-01", "bop-jk-02")
    assert corridor is not None
    assert corridor["expected_transit_sec"]["optimal"] == 25.0
    print("\n[Topology Load PASS] Graph initialized with bidirectional corridors.")


def test_spatial_affinity_gaussian_window():
    """Verify Gaussian spatial affinity scoring around optimal transit time."""
    top = CameraTopology()
    top.register_corridor("cam_a", "cam_b", min_sec=10.0, optimal_sec=30.0, max_sec=90.0)

    # 1. At optimal transit time (30s) -> Peak affinity
    score_opt = top.compute_spatial_affinity("cam_a", "cam_b", dt_sec=30.0)
    assert score_opt >= 0.95, f"Expected peak affinity >= 0.95 at t_optimal, got {score_opt}"

    # 2. Within active window (45s) -> High affinity
    score_window = top.compute_spatial_affinity("cam_a", "cam_b", dt_sec=45.0)
    assert score_window >= 0.70, f"Expected high affinity in window, got {score_window}"

    # 3. Impossibly fast (< min_sec, e.g. 2s) -> Heavy penalty
    score_fast = top.compute_spatial_affinity("cam_a", "cam_b", dt_sec=2.0)
    assert score_fast < 0.25, f"Expected penalty for impossibly fast transit, got {score_fast}"

    # 4. Long delayed transit (300s) -> Smooth decay
    score_delay = top.compute_spatial_affinity("cam_a", "cam_b", dt_sec=300.0)
    assert score_delay < 0.40, f"Expected decay for delayed transit, got {score_delay}"
    print(f"\n[Affinity PASS] Optimal: {score_opt:.2f} | Window: {score_window:.2f} | Fast: {score_fast:.2f} | Delay: {score_delay:.2f}")


def test_fused_scoring_rescues_lighting_variation():
    """
    CRITICAL TEST: Verify that a legitimate person whose visual Re-ID score drops to 0.60
    (below raw 0.65 threshold) due to glare/angle is SAVED from an ID switch because they
    arrived exactly on time via an active corridor.
    """
    top = CameraTopology()
    top.register_corridor("post_1", "post_2", min_sec=5.0, optimal_sec=20.0, max_sec=60.0)

    gtm = GlobalTraceManager(
        similarity_threshold=0.65,
        reid_weight=0.70,
        spatial_weight=0.30,
        topology=top,
    )
    # Register physical coordinates
    gtm.register_camera_coordinates("post_1", 32.7240, 74.6710)
    gtm.register_camera_coordinates("post_2", 32.7243, 74.6710)  # ~33m apart

    # Original person crop on post_1
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame1[50:250, 100:200, 0] = 220
    frame1[250:400, 100:200, :] = 30

    t0 = 1700000000.0
    tid1, _, _ = gtm.update_track(
        camera_id="post_1",
        camera_name="Post 1",
        local_track_id=1,
        box=[100, 50, 200, 400],
        frame_bgr=frame1,
        confidence=0.92,
        timestamp=t0,
    )
    assert tid1 == "TRC-0001"

    # In post_2, lighting changes significantly (visual similarity drops to ~0.60)
    frame2 = np.clip(frame1.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    # Add severe tint to pull appearance score down
    frame2[50:250, 100:200, 1] = 120

    # Person arrives at post_2 exactly 20.0 seconds later (optimal corridor time!)
    t1 = t0 + 20.0
    tid2, is_cross2, fused_score = gtm.update_track(
        camera_id="post_2",
        camera_name="Post 2",
        local_track_id=14,
        box=[120, 60, 220, 410],
        frame_bgr=frame2,
        confidence=0.90,
        timestamp=t1,
    )

    # Must be successfully merged into TRC-0001!
    assert tid2 == "TRC-0001", f"Corridor expectation must rescue borderline ReID! Got {tid2}"
    assert is_cross2 is True
    assert fused_score >= 0.65, f"Fused score should be >= 0.65, got {fused_score}"

    trace = gtm.get_trace("TRC-0001")
    wp = trace.waypoints[-1]
    assert "spatial_score" in wp
    assert "fused_score" in wp
    print(f"\n[PASS] Borderline appearance rescued by corridor! Fused={fused_score:.3f} (Spatial={wp['spatial_score']:.2f}, ReID={wp['reid_score']:.2f})")


def test_fused_scoring_penalizes_improbable_transit():
    """
    CRITICAL TEST: Verify that a person with similar clothes (ReID = 0.68) arriving
    outside corridor expectations is penalized and REJECTED from false merging.
    """
    top = CameraTopology()
    # Explicit corridor with min 30s, optimal 60s
    top.register_corridor("zone_a", "zone_b", min_sec=30.0, optimal_sec=60.0, max_sec=180.0)

    gtm = GlobalTraceManager(
        similarity_threshold=0.65,
        reid_weight=0.70,
        spatial_weight=0.30,
        topology=top,
    )
    gtm.register_camera_coordinates("zone_a", 32.7240, 74.6710)
    gtm.register_camera_coordinates("zone_b", 32.7246, 74.6710)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:250, 100:200, 1] = 210
    frame[250:400, 100:200, :] = 40

    t0 = 1700000000.0
    gtm.update_track("zone_a", "Zone A", 1, [100, 50, 200, 400], frame, 0.9, timestamp=t0)

    # Person arrives at zone_b after only 3.0s (underflow against min_sec=30s)
    t1 = t0 + 3.0
    tid2, is_cross2, fused_score = gtm.update_track(
        "zone_b", "Zone B", 99, [100, 50, 200, 400], frame, 0.9, timestamp=t1
    )

    # Spatial penalty pulls fused score below 0.65 threshold -> Assigned new trace TRC-0002!
    assert tid2 == "TRC-0002", f"Improbable transit must be rejected and initialized as TRC-0002! Got {tid2}"
    assert is_cross2 is False
    print(f"\n[PASS] Improbable transit successfully rejected by spatial penalty (Assigned {tid2}).")


def test_online_transition_learning():
    """Verify that verified transitions adaptively update the optimal transit time."""
    top = CameraTopology()
    top.register_corridor("gate_x", "gate_y", min_sec=5.0, optimal_sec=30.0, max_sec=90.0)

    initial_opt = top.get_corridor("gate_x", "gate_y")["expected_transit_sec"]["optimal"]
    assert initial_opt == 30.0

    # Repeated transits averaging 18.0 seconds
    for _ in range(5):
        top.record_transition("gate_x", "gate_y", dt_sec=18.0)

    new_opt = top.get_corridor("gate_x", "gate_y")["expected_transit_sec"]["optimal"]
    assert new_opt < initial_opt, f"Optimal time should adapt downwards toward 18.0s, got {new_opt}"
    assert top.get_corridor("gate_x", "gate_y")["transit_count"] == 5
    print(f"\n[Online Learning PASS] Optimal transit adapted: 30.0s -> {new_opt:.1f}s after 5 observations.")


def test_rest_api_topology_endpoints():
    """Verify GET /api/tracking/topology and POST /api/tracking/topology/corridor."""
    client = TestClient(app)

    # 1. GET topology
    resp = client.get("/api/tracking/topology")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "topology" in data
    assert data["topology"]["total_corridors"] >= 1

    # 2. POST corridor
    post_resp = client.post(
        "/api/tracking/topology/corridor",
        data={
            "from_camera": "test_cam_1",
            "to_camera": "test_cam_2",
            "min_transit_sec": 4.0,
            "optimal_transit_sec": 18.0,
            "max_transit_sec": 80.0,
            "distance_m": 35.0,
            "bidirectional": True,
        },
    )
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["status"] == "success"
    assert post_data["corridor"]["from_camera"] == "test_cam_1"
    assert post_data["corridor"]["expected_transit_sec"]["optimal"] == 18.0
    print("\n[REST API PASS] Topology endpoints verified successfully.")


if __name__ == "__main__":
    print("\n=======================================================")
    print(" Running Camera Topology & Transition Matrix Tests ")
    print("=======================================================\n")
    test_camera_topology_graph_initialization()
    print(" [1/6] test_camera_topology_graph_initialization: PASSED")
    test_spatial_affinity_gaussian_window()
    print(" [2/6] test_spatial_affinity_gaussian_window: PASSED")
    test_fused_scoring_rescues_lighting_variation()
    print(" [3/6] test_fused_scoring_rescues_lighting_variation: PASSED")
    test_fused_scoring_penalizes_improbable_transit()
    print(" [4/6] test_fused_scoring_penalizes_improbable_transit: PASSED")
    test_online_transition_learning()
    print(" [5/6] test_online_transition_learning: PASSED")
    test_rest_api_topology_endpoints()
    print(" [6/6] test_rest_api_topology_endpoints: PASSED")
    print("\n ALL 6 CAMERA TOPOLOGY & TRANSITION TESTS PASSED! \n")
