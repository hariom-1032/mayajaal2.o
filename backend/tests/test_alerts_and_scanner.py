import hashlib
from pathlib import Path
import sys
import time

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.pipeline.alert_service import alert_service, SNAPSHOTS_DIR
from app.pipeline.feed_scanner import feed_scanner_service



def test_alert_service_lifecycle():
    """Test AlertService creation, filtering, acknowledgement, and ring buffer."""
    # Create test alert
    alert = alert_service.create_alert(
        title="🚨 TEST SUSPECT SIGHTING",
        location="Camera 01 — Gate Alpha",
        severity="High",
        camera_id="cam-01",
        camera_name="Gate Alpha Optical",
        class_name="suspect",
        confidence=0.92,
        suspect_name="Test Suspect",
        threat_level="CRITICAL",
        category="Terrorism / Infiltration",
        notes="Automated test alert",
    )

    assert alert["id"].startswith("alert-")
    assert alert["suspectName"] == "Test Suspect"
    assert alert["threatLevel"] == "CRITICAL"
    assert alert["acknowledged"] is False

    # Retrieve with filter
    results = alert_service.get_alerts(severity="High", camera_id="cam-01")
    assert any(a["id"] == alert["id"] for a in results)

    # Acknowledge
    ack_res = alert_service.acknowledge_alert(alert["id"])
    assert ack_res is True

    # Verify acknowledged state
    updated_alerts = alert_service.get_alerts(acknowledged=True)
    assert any(a["id"] == alert["id"] for a in updated_alerts)

    print("[PASS] AlertService creation, filtering, and acknowledgement verified.")


def test_continuous_face_scanner_telemetry():
    """Test ContinuousFaceScanner status and start/stop controls."""
    status_initial = feed_scanner_service.get_status()
    assert "running" in status_initial
    assert "worker_count" in status_initial
    assert "sampling_interval_sec" in status_initial

    # Test scanner start
    start_res = feed_scanner_service.start()
    assert start_res["success"] is True

    time.sleep(0.5)

    status_running = feed_scanner_service.get_status()
    assert status_running["running"] is True

    # Test scanner stop
    stop_res = feed_scanner_service.stop()
    assert stop_res["success"] is True

    status_stopped = feed_scanner_service.get_status()
    assert status_stopped["running"] is False
    print("[PASS] ContinuousFaceScanner telemetry, start, and stop lifecycle verified.")


def test_alerts_api_endpoints():
    """Test the REST API endpoints in alerts_router using FastAPI TestClient."""
    client = TestClient(app)

    # 1. GET /api/alerts
    resp = client.get("/api/alerts")
    assert resp.status_code == 200
    alerts = resp.json()
    assert isinstance(alerts, list)
    print(f"[PASS] GET /api/alerts returned 200 OK with {len(alerts)} alerts.")

    # 2. POST /api/alerts (Manual dispatch)
    new_alert_payload = {
        "title": "🚨 DISPATCH: TEST BOLO DETECTED",
        "location": "Sector 4 — Ridge Cam",
        "severity": "High",
        "cameraId": "cam-02",
        "cameraName": "Ridge Cam",
        "className": "suspect",
        "confidence": 0.89,
        "suspectName": "Ramesh Kumar",
        "threatLevel": "HIGH",
        "category": "Wanted / BOLO",
        "notes": "Spotted near secondary fence",
    }
    resp = client.post("/api/alerts", json=new_alert_payload)
    assert resp.status_code == 201
    created = resp.json()
    assert created["suspectName"] == "Ramesh Kumar"
    created_id = created["id"]
    print("[PASS] POST /api/alerts successfully created alert.")

    # 3. PATCH /api/alerts/{alert_id}/acknowledge
    resp = client.patch(f"/api/alerts/{created_id}/acknowledge")
    assert resp.status_code == 200
    ack_data = resp.json()
    assert ack_data["acknowledged"] is True
    print("[PASS] PATCH /api/alerts/{id}/acknowledge marked alert as acknowledged.")

    # 4. GET /api/alerts/scanner/status
    resp = client.get("/api/alerts/scanner/status")
    assert resp.status_code == 200
    st = resp.json()
    assert "running" in st
    print("[PASS] GET /api/alerts/scanner/status returned scanner telemetry.")

    # 5. Snapshot test: create dummy test snapshot and retrieve via API
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_name = "test_snapshot_api.jpg"
    dummy_path = SNAPSHOTS_DIR / dummy_name
    cv2.imwrite(str(dummy_path), dummy_img)

    resp = client.get(f"/api/alerts/snapshots/{dummy_name}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    print("[PASS] GET /api/alerts/snapshots/{filename} served JPEG snapshot.")

    # Clean up dummy image
    if dummy_path.exists():
        dummy_path.unlink()


def test_suspect_trajectory_and_qrt_dispatch():
    """Test multi-camera suspect trajectory reconstruction and QRT team dispatch."""
    client = TestClient(app)

    # Dispatch two sequential sightings for suspect "Vikram Singh"
    t1 = int(time.time() * 1000) - 600000  # 10 mins ago
    t2 = int(time.time() * 1000) - 60000   # 1 min ago

    alert1 = alert_service.create_alert(
        title="🚨 SUSPECT: Vikram Singh [CRITICAL]",
        location="RS Pura BOP Alpha — Zero Line Fence",
        severity="High",
        camera_id="bop-jk-01",
        camera_name="RS Pura BOP Alpha",
        class_name="suspect",
        confidence=0.94,
        suspect_name="Vikram Singh",
        threat_level="CRITICAL",
        category="Terrorism / Infiltration",
        notes="First border sighting",
    )
    # Manually adjust timestamp for chronological delta
    alert1["timestamp"] = t1
    alert_service._save_alerts()

    alert2 = alert_service.create_alert(
        title="🚨 SUSPECT: Vikram Singh [CRITICAL]",
        location="Suchetgarh JCP Octroi Gate",
        severity="High",
        camera_id="bop-jk-02",
        camera_name="Suchetgarh Octroi Gate",
        class_name="suspect",
        confidence=0.91,
        suspect_name="Vikram Singh",
        threat_level="CRITICAL",
        category="Terrorism / Infiltration",
        notes="Second sighting moving towards highway",
    )
    alert2["timestamp"] = t2
    alert_service._save_alerts()

    # 1. GET /api/alerts/trajectory/{suspect_name}
    resp = client.get("/api/alerts/trajectory/Vikram Singh")
    assert resp.status_code == 200
    traj = resp.json()
    assert traj["found"] is True
    assert traj["suspect_name"] == "Vikram Singh"
    assert traj["total_sightings"] >= 2
    assert len(traj["waypoints"]) >= 2
    assert traj["total_distance_km"] > 0
    print(f"[PASS] GET /api/alerts/trajectory returned {traj['total_sightings']} waypoints, total {traj['total_distance_km']} km.")

    # 2. POST /api/alerts/{alert_id}/dispatch (QRT deployment)
    dispatch_payload = {
        "unitName": "QRT Strike Force Bravo",
        "notes": "Intercept suspect heading northeast",
    }
    resp = client.post(f"/api/alerts/{alert2['id']}/dispatch", json=dispatch_payload)
    assert resp.status_code == 200
    dispatch_res = resp.json()
    assert dispatch_res["success"] is True
    assert dispatch_res["dispatch"]["unit"] == "QRT Strike Force Bravo"
    assert dispatch_res["dispatch"]["status"] == "Dispatched"
    print("[PASS] POST /api/alerts/{alert_id}/dispatch successfully deployed QRT unit.")


def cleanup_test_data():
    """Clear all test alerts and restore camera statuses to 'online' so no dummy data remains."""
    alert_service.clear_all()
    cameras_file = backend_dir / "app" / "data" / "cameras.json"
    if cameras_file.exists():
        try:
            import json
            with open(cameras_file, "r", encoding="utf-8") as f:
                cams = json.load(f)
            updated = False
            for c in cams:
                if c.get("id") in ("bop-jk-01", "bop-jk-02") and c.get("status") == "alert":
                    c["status"] = "online"
                    updated = True
            if updated:
                with open(cameras_file, "w", encoding="utf-8") as f:
                    json.dump(cams, f, indent=2)
        except Exception:
            pass


if __name__ == "__main__":
    print("\n--- Running Alerts, Scanner, and Trajectory Test Suite ---")
    try:
        test_alert_service_lifecycle()
        test_continuous_face_scanner_telemetry()
        test_alerts_api_endpoints()
        test_suspect_trajectory_and_qrt_dispatch()
        print("\n===================================================================")
        print(" ALL TESTS PASSED: PHASE 3 TACTICAL TRAJECTORY & QRT DISPATCH ")
        print("===================================================================\n")
    finally:
        cleanup_test_data()


