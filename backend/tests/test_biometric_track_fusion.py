"""
test_biometric_track_fusion.py — Unit & Integration Test Suite for Phase 4:
Biometric Track Association & Retrospective History Resolution.

Tests:
1. Spatial 2D Face-to-Person Bounding Box containment & IoF matching.
2. Anonymous multi-camera transit across cameras 1 and 2.
3. Facial recognition identification at camera 3 (choke point).
4. Retrospective backfill of suspect attribution to past waypoints.
5. Trajectory reconstruction preserving retrospective biometric audit trail.
6. REST API endpoint POST /api/tracking/global/traces/{trace_id}/resolve and query filters.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tracking.global_tracker import (
    GlobalTraceManager,
    GlobalTrace,
    match_face_to_person_track,
)
from app.tracking.topology import CameraTopology


def make_dummy_feature(seed: int, dim: int = 512) -> np.ndarray:
    """Generate normalized dummy Re-ID feature vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_face_to_person_box_matching():
    """Verify that match_face_to_person_track correctly pairs face bounding boxes to person bboxes."""
    person_tracks = [
        {
            "track_id": 101,
            "class_name": "person",
            "class_id": 0,
            "box": [100.0, 100.0, 200.0, 350.0],  # Person A: width=100, height=250, upper body y in [100, 262.5]
            "global_trace_id": "TRC-0001",
        },
        {
            "track_id": 102,
            "class_name": "person",
            "class_id": 0,
            "box": [400.0, 120.0, 500.0, 360.0],  # Person B: far to the right
            "global_trace_id": "TRC-0002",
        },
    ]

    # Face 1: located directly on Person A's head/face [130, 110, 170, 160]
    face_a = [130.0, 110.0, 170.0, 160.0]
    matched_a = match_face_to_person_track(face_a, person_tracks)
    assert matched_a is not None, "Face A must match Person A"
    assert matched_a["track_id"] == 101
    assert matched_a["global_trace_id"] == "TRC-0001"

    # Face 2: located on Person B's head/face [425, 130, 475, 185]
    face_b = [425.0, 130.0, 475.0, 185.0]
    matched_b = match_face_to_person_track(face_b, person_tracks)
    assert matched_b is not None, "Face B must match Person B"
    assert matched_b["track_id"] == 102
    assert matched_b["global_trace_id"] == "TRC-0002"

    # Face 3: located in empty space [600, 50, 650, 100]
    face_empty = [600.0, 50.0, 650.0, 100.0]
    matched_none = match_face_to_person_track(face_empty, person_tracks)
    assert matched_none is None, "Face in empty space must not match any person track"


def test_anonymous_transit_and_retrospective_suspect_lock():
    """
    Verify complete Phase 4 workflow:
    1. Anonymous person enters Camera 1 (TRC-0001, is_suspect=False).
    2. Transits to Camera 2 via corridor (waypoint 2, still anonymous).
    3. Facial recognition locks suspect identity at Camera 3.
    4. Past waypoints (Cam 1 & Cam 2) are retrospectively backfilled with suspect details.
    """
    gtm = GlobalTraceManager(
        similarity_threshold=0.60,
        max_pedestrian_speed_kmh=30.0,
        spatial_collocation_threshold_m=10.0,
    )
    # Set coordinates for 3 sequential cameras along a perimeter corridor
    gtm.set_camera_coordinates("cam_01", 32.7240, 74.8560)
    gtm.set_camera_coordinates("cam_02", 32.7242, 74.8562)  # ~30m from cam_01
    gtm.set_camera_coordinates("cam_03", 32.7245, 74.8565)  # ~40m from cam_02

    feature = make_dummy_feature(seed=42)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # ── Step 1: Camera 1 Sighting (Anonymous entry) ──
    t1 = 1000.0
    trace_id_1, is_cross_1, _ = gtm.update_track(
        camera_id="cam_01",
        camera_name="Cam 01 - Perimeter Gate",
        local_track_id=1,
        box=[100, 100, 200, 350],
        frame_bgr=fake_frame,
        confidence=0.88,
        suspect_name=None,  # Unknown person
        threat_level=None,
        timestamp=t1,
    )
    # Manually inject the normalized feature into the initialized trace for deterministic testing
    trace_1 = gtm.get_trace(trace_id_1)
    trace_1.feature = feature
    trace_1.feature_history = [feature]

    assert trace_id_1 == "TRC-0001"
    assert not trace_1.is_suspect, "Initial trace must be anonymous"
    assert not trace_1.biometric_locked, "Initial trace must not be biometric locked"
    assert len(trace_1.waypoints) == 1
    assert not trace_1.waypoints[0]["is_suspect"]
    assert not trace_1.waypoints[0].get("retrospective", False)

    # ── Step 2: Camera 2 Sighting (Corridor transit handoff, still anonymous) ──
    t2 = t1 + 15.0  # 15 seconds transit time
    trace_id_2, is_cross_2, fused_sc_2 = gtm.update_track(
        camera_id="cam_02",
        camera_name="Cam 02 - Transit Corridor",
        local_track_id=5,
        box=[110, 120, 210, 360],
        frame_bgr=fake_frame,
        confidence=0.85,
        suspect_name=None,  # Still unknown
        timestamp=t2,
    )
    # Associate appearance
    trace_2 = gtm.get_trace(trace_id_1)
    trace_2.add_waypoint(
        camera_id="cam_02",
        camera_name="Cam 02 - Transit Corridor",
        box=[110, 120, 210, 360],
        confidence=0.85,
        reid_score=0.88,
        timestamp=t2,
        spatial_score=0.95,
        fused_score=0.90,
        speed_kmh=7.2,
    )
    assert len(trace_2.waypoints) == 2
    assert not trace_2.is_suspect
    assert not trace_2.waypoints[0]["is_suspect"]
    assert not trace_2.waypoints[1]["is_suspect"]

    # ── Step 3: Camera 3 Sighting — Choke Point Facial Recognition Match ──
    # High-confidence face recognition matches "Tariq Aziz", Threat CRITICAL
    t3 = t2 + 20.0  # 20 seconds later
    resolved_trace = gtm.resolve_trace_suspect(
        global_track_id=trace_id_1,
        suspect_name="Tariq Aziz",
        threat_level="CRITICAL",
        category="WANTED_TERRORIST",
        face_confidence=0.94,
        source_camera_id="cam_03",
        timestamp=t3,
    )

    assert resolved_trace is not None
    assert resolved_trace.is_suspect is True
    assert resolved_trace.suspect_name == "Tariq Aziz"
    assert resolved_trace.threat_level == "CRITICAL"
    assert resolved_trace.biometric_locked is True
    assert resolved_trace.biometric_confidence == 0.94
    assert resolved_trace.biometric_source_camera == "cam_03"
    assert resolved_trace.resolved_retrospectively is True

    # ── Step 4: Verify Retrospective Backfill on Past Waypoints ──
    wp1 = resolved_trace.waypoints[0]
    wp2 = resolved_trace.waypoints[1]

    # Cam 1 waypoint must now be retrospectively tagged
    assert wp1["camera_id"] == "cam_01"
    assert wp1["is_suspect"] is True
    assert wp1["suspect_name"] == "Tariq Aziz"
    assert wp1["threat_level"] == "CRITICAL"
    assert wp1["biometric_locked"] is True
    assert wp1["retrospective"] is True, "Waypoint 1 was recorded prior to face identification -> must be retrospective"

    # Cam 2 waypoint must also be retrospectively tagged
    assert wp2["camera_id"] == "cam_02"
    assert wp2["is_suspect"] is True
    assert wp2["suspect_name"] == "Tariq Aziz"
    assert wp2["threat_level"] == "CRITICAL"
    assert wp2["biometric_locked"] is True
    assert wp2["retrospective"] is True, "Waypoint 2 was recorded prior to face identification -> must be retrospective"

    # ── Step 5: Add subsequent Waypoint 3 at Cam 3 (Active Recognition) ──
    resolved_trace.add_waypoint(
        camera_id="cam_03",
        camera_name="Cam 03 - Checkpoint Gate",
        box=[120, 130, 220, 370],
        confidence=0.94,
        reid_score=0.92,
        timestamp=t3,
        spatial_score=0.90,
        fused_score=0.91,
        speed_kmh=6.5,
    )
    wp3 = resolved_trace.waypoints[2]
    assert wp3["camera_id"] == "cam_03"
    assert wp3["is_suspect"] is True
    assert wp3["suspect_name"] == "Tariq Aziz"
    assert wp3["retrospective"] is False, "Waypoint 3 occurred during/after biometric identification -> not retrospective"


def test_trajectory_retrospective_metadata():
    """Verify get_trace_trajectory includes all biometric locking and retrospective audit details."""
    gtm = GlobalTraceManager()
    gtm.set_camera_coordinates("cam_01", 32.7240, 74.8560)
    gtm.set_camera_coordinates("cam_02", 32.7250, 74.8570)

    fake_feature = make_dummy_feature(seed=101)
    trace = GlobalTrace(
        trace_id="TRC-0099",
        feature=fake_feature,
        camera_id="cam_01",
        camera_name="Camera Alpha",
        box=[50, 50, 150, 300],
        confidence=0.90,
        timestamp=2000.0,
    )
    gtm._traces["TRC-0099"] = trace
    trace.add_waypoint(
        camera_id="cam_02",
        camera_name="Camera Beta",
        box=[60, 60, 160, 310],
        confidence=0.88,
        reid_score=0.85,
        timestamp=2030.0,
    )

    # Retrospectively resolve suspect
    gtm.resolve_trace_suspect(
        global_track_id="TRC-0099",
        suspect_name="Carlos Mendez",
        threat_level="HIGH",
        category="BOLO_NARCOTICS",
        face_confidence=0.91,
        source_camera_id="cam_02",
        timestamp=2030.0,
    )

    traj = gtm.get_trace_trajectory("TRC-0099")
    assert traj["found"] is True
    assert traj["suspect_name"] == "Carlos Mendez"
    assert traj["threat_level"] == "HIGH"
    assert traj["biometric_locked"] is True
    assert traj["resolved_retrospectively"] is True
    assert traj["biometric_confidence"] == 0.91
    assert traj["biometric_source_camera"] == "cam_02"

    wps = traj["waypoints"]
    assert len(wps) == 2
    assert wps[0]["is_suspect"] is True
    assert wps[0]["suspect_name"] == "Carlos Mendez"
    assert wps[0]["retrospective"] is True
    assert wps[1]["retrospective"] is True


def test_rest_api_trace_resolution_and_filters():
    """Test REST API endpoint POST /api/tracking/global/traces/{trace_id}/resolve and GET filters."""
    client = TestClient(app)

    # 1. Reset traces
    res = client.delete("/api/tracking/global/traces")
    assert res.status_code == 200

    # 2. Seed an anonymous trace via global_trace_manager directly
    from app.tracking.global_tracker import global_trace_manager
    fake_feat = make_dummy_feature(seed=202)
    trace = GlobalTrace(
        trace_id="TRC-0001",
        feature=fake_feat,
        camera_id="cam_01",
        camera_name="North Gate Cam",
        box=[100, 100, 200, 350],
        confidence=0.82,
        timestamp=3000.0,
    )
    global_trace_manager._traces["TRC-0001"] = trace

    # Check listing before resolution
    res_list = client.get("/api/tracking/global/traces")
    assert res_list.status_code == 200
    traces = res_list.json()["traces"]
    assert len(traces) == 1
    assert traces[0]["trace_id"] == "TRC-0001"
    assert traces[0]["is_suspect"] is False
    assert traces[0]["biometric_locked"] is False

    # Check suspect-only filter -> should return 0
    res_suspects = client.get("/api/tracking/global/traces?suspect_only=true")
    assert len(res_suspects.json()["traces"]) == 0

    # 3. Call POST /api/tracking/global/traces/TRC-0001/resolve
    resolve_payload = {
        "suspect_name": "Hamid Al-Sayed",
        "threat_level": "CRITICAL",
        "category": "INFILTRATION",
        "face_confidence": 0.96,
        "source_camera_id": "checkpoint_cam_03",
    }
    res_resolve = client.post("/api/tracking/global/traces/TRC-0001/resolve", json=resolve_payload)
    assert res_resolve.status_code == 200
    data = res_resolve.json()
    assert data["status"] == "success"
    assert data["trace"]["suspect_name"] == "Hamid Al-Sayed"
    assert data["trace"]["threat_level"] == "CRITICAL"
    assert data["trace"]["biometric_locked"] is True
    assert data["trace"]["resolved_retrospectively"] is True
    assert data["retrospective_waypoints_updated"] == 1

    # 4. Verify suspect-only and biometric_locked filters now return TRC-0001
    res_suspects_after = client.get("/api/tracking/global/traces?suspect_only=true")
    assert len(res_suspects_after.json()["traces"]) == 1
    assert res_suspects_after.json()["traces"][0]["suspect_name"] == "Hamid Al-Sayed"

    res_biometric = client.get("/api/tracking/global/traces?biometric_locked_only=true")
    assert len(res_biometric.json()["traces"]) == 1
    assert res_biometric.json()["traces"][0]["biometric_locked"] is True

    # 5. Check 404 for nonexistent trace
    res_404 = client.post("/api/tracking/global/traces/TRC-9999/resolve", json=resolve_payload)
    assert res_404.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
