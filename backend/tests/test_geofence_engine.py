import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from app.pipeline.geofence_engine import (
    is_point_in_polygon,
    ccw,
    check_tripwire_crossing,
    compute_footpoint,
    GeofenceEngine,
)


class TestGeofenceEngine(unittest.TestCase):

    def test_point_in_polygon_square(self):
        # Square polygon from (0.2, 0.2) to (0.8, 0.8)
        poly = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

        self.assertTrue(is_point_in_polygon(0.5, 0.5, poly))
        self.assertTrue(is_point_in_polygon(0.3, 0.7, poly))
        self.assertFalse(is_point_in_polygon(0.1, 0.5, poly))
        self.assertFalse(is_point_in_polygon(0.9, 0.5, poly))
        self.assertFalse(is_point_in_polygon(0.5, 0.1, poly))
        self.assertFalse(is_point_in_polygon(0.5, 0.9, poly))

    def test_point_in_polygon_concave_l_shape(self):
        # L-shaped polygon
        poly = [
            [0.0, 0.0],
            [0.4, 0.0],
            [0.4, 0.4],
            [1.0, 0.4],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
        self.assertTrue(is_point_in_polygon(0.2, 0.2, poly))
        self.assertTrue(is_point_in_polygon(0.2, 0.8, poly))
        self.assertTrue(is_point_in_polygon(0.8, 0.8, poly))
        # Cutout (0.7, 0.2) is outside
        self.assertFalse(is_point_in_polygon(0.7, 0.2, poly))

    def test_compute_footpoint(self):
        # Bounding box [x1, y1, x2, y2]
        box = [100.0, 50.0, 200.0, 150.0]
        # Frame resolution (1000, 1000)
        fp = compute_footpoint(box, frame_resolution=(1000, 1000))
        self.assertAlmostEqual(fp[0], 0.15, places=4)
        self.assertAlmostEqual(fp[1], 0.15, places=4)

        # Normalized input box [0.2, 0.1, 0.4, 0.9]
        norm_box = [0.2, 0.1, 0.4, 0.9]
        fp_norm = compute_footpoint(norm_box)
        self.assertAlmostEqual(fp_norm[0], 0.3, places=4)
        self.assertAlmostEqual(fp_norm[1], 0.9, places=4)

    def test_tripwire_crossing_forward_and_reverse(self):
        # Horizontal tripwire from left (0.1, 0.5) to right (0.9, 0.5)
        # Tripwire vector W = (0.8, 0.0). Normal N = (-0.0, 0.8) -> points DOWN.
        p1 = [0.1, 0.5]
        p2 = [0.9, 0.5]

        # Target moving top-to-bottom: (0.5, 0.2) -> (0.5, 0.8)
        prev_down = (0.5, 0.2)
        curr_down = (0.5, 0.8)

        # Moving downward aligns with N (vy = +0.6 > 0), so FORWARD
        crossed_fwd, dir_fwd, _ = check_tripwire_crossing(p1, p2, prev_down, curr_down, direction="FORWARD")
        self.assertTrue(crossed_fwd)
        self.assertEqual(dir_fwd, "FORWARD")

        # Moving upward is REVERSE
        prev_up = (0.5, 0.8)
        curr_up = (0.5, 0.2)
        crossed_rev, dir_rev, _ = check_tripwire_crossing(p1, p2, prev_up, curr_up, direction="REVERSE")
        self.assertTrue(crossed_rev)
        self.assertEqual(dir_rev, "REVERSE")

        # When configured as FORWARD, an upward move should NOT trigger
        crossed_mismatch, _, _ = check_tripwire_crossing(p1, p2, prev_up, curr_up, direction="FORWARD")
        self.assertFalse(crossed_mismatch)

        # Bidirectional triggers on both directions
        crossed_bi_1, _, _ = check_tripwire_crossing(p1, p2, prev_down, curr_down, direction="BIDIRECTIONAL")
        crossed_bi_2, _, _ = check_tripwire_crossing(p1, p2, prev_up, curr_up, direction="BIDIRECTIONAL")
        self.assertTrue(crossed_bi_1)
        self.assertTrue(crossed_bi_2)

    def test_tripwire_no_intersection(self):
        p1 = [0.1, 0.5]
        p2 = [0.9, 0.5]

        # Target moving parallel without intersecting: (0.5, 0.2) -> (0.5, 0.4)
        crossed, _, _ = check_tripwire_crossing(p1, p2, (0.5, 0.2), (0.5, 0.4), direction="FORWARD")
        self.assertFalse(crossed)

        # Target moving completely to the right of the segment
        crossed_far, _, _ = check_tripwire_crossing(p1, p2, (0.95, 0.2), (0.95, 0.8), direction="FORWARD")
        self.assertFalse(crossed_far)

    def test_engine_crud_and_evaluate(self):
        engine = GeofenceEngine()

        test_cam = "test-cam-eval"
        zone = engine.add_zone({
            "name": "Unit Test Zone",
            "cameraId": test_cam,
            "severity": "HIGH",
            "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            "targetClasses": ["person", "suspect"],
            "cooldownSeconds": 60.0,
        })
        self.assertTrue(zone["id"].startswith("zone-"))

        tripwire = engine.add_tripwire({
            "name": "Unit Test Tripwire",
            "cameraId": test_cam,
            "p1": [0.1, 0.5],
            "p2": [0.9, 0.5],
            "direction": "FORWARD",
            "targetClasses": ["person"],
            "cooldownSeconds": 60.0,
        })
        self.assertTrue(tripwire["id"].startswith("wire-"))

        # Frame 1: Track #101 at (0.5, 0.1) - outside polygon and above wire
        breaches_f1 = engine.evaluate_tracks(
            camera_id=test_cam,
            tracked_objects=[{
                "track_id": 101,
                "box": [0.45, 0.05, 0.55, 0.1],
                "class_name": "person",
                "confidence": 0.92,
            }],
            auto_alert=False,
        )
        self.assertEqual(len(breaches_f1), 0)

        # Frame 2: Track #101 moves to (0.5, 0.6) - crosses tripwire downward AND enters polygon
        breaches_f2 = engine.evaluate_tracks(
            camera_id=test_cam,
            tracked_objects=[{
                "track_id": 101,
                "box": [0.45, 0.5, 0.55, 0.6],
                "class_name": "person",
                "confidence": 0.95,
            }],
            auto_alert=False,
        )

        types = [b["type"] for b in breaches_f2]
        self.assertIn("zone_breach", types)
        self.assertIn("tripwire_crossing", types)

        # Cleanup test items
        self.assertTrue(engine.delete_zone(zone["id"]))
        self.assertTrue(engine.delete_tripwire(tripwire["id"]))

    def test_access_control_authorized_silent_pass(self):
        """Verify that an authorized personnel entering polygon triggers NO breach/alert."""
        engine = GeofenceEngine()
        test_cam = "cam-test-access-ctrl"
        zone = engine.add_zone({
            "name": "Access Controlled Zone",
            "cameraId": test_cam,
            "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            "targetClasses": ["person", "suspect"],
            "accessMode": "AUTHORIZED_PASS",
            "cooldownSeconds": 10.0,
        })

        # Authorized employee enters polygon
        breaches = engine.evaluate_tracks(
            camera_id=test_cam,
            tracked_objects=[{
                "track_id": 201,
                "box": [0.45, 0.5, 0.55, 0.6],
                "class_name": "person",
                "confidence": 0.98,
                "person_name": "Hariom (Staff)",
                "identity_status": "authorized",
                "is_authorized": True,
                "is_suspect": False,
                "is_unknown": False,
            }],
            auto_alert=False,
        )

        # MUST BE 0 - Silent Pass
        self.assertEqual(len(breaches), 0, "Authorized person should pass silently without generating any breaches")
        engine.delete_zone(zone["id"])

    def test_access_control_suspect_critical_alert(self):
        """Verify that a flagged suspect entering polygon triggers a CRITICAL suspect alert."""
        engine = GeofenceEngine()
        test_cam = "cam-test-access-ctrl"
        zone = engine.add_zone({
            "name": "High Security Zone",
            "cameraId": test_cam,
            "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            "targetClasses": ["person", "suspect"],
            "accessMode": "AUTHORIZED_PASS",
            "cooldownSeconds": 10.0,
        })

        # Suspect enters polygon
        breaches = engine.evaluate_tracks(
            camera_id=test_cam,
            tracked_objects=[{
                "track_id": 301,
                "box": [0.45, 0.5, 0.55, 0.6],
                "class_name": "suspect",
                "confidence": 0.95,
                "suspect_name": "Raja Waseem",
                "identity_status": "suspect",
                "is_authorized": False,
                "is_suspect": True,
                "is_unknown": False,
            }],
            auto_alert=False,
        )

        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0]["alertType"], "suspect_breach")
        self.assertEqual(breaches[0]["severity"], "CRITICAL")
        self.assertEqual(breaches[0]["suspectName"], "Raja Waseem")
        engine.delete_zone(zone["id"])

    def test_access_control_unknown_intruder_alert(self):
        """Verify that an unknown/unregistered person triggers an unknown intrusion alert."""
        engine = GeofenceEngine()
        test_cam = "cam-test-access-ctrl"
        zone = engine.add_zone({
            "name": "Perimeter Grid Alpha",
            "cameraId": test_cam,
            "polygon": [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            "targetClasses": ["person", "suspect"],
            "accessMode": "AUTHORIZED_PASS",
            "cooldownSeconds": 10.0,
        })

        # Unknown person enters polygon
        breaches = engine.evaluate_tracks(
            camera_id=test_cam,
            tracked_objects=[{
                "track_id": 401,
                "box": [0.45, 0.5, 0.55, 0.6],
                "class_name": "person",
                "confidence": 0.88,
                "identity_status": "unknown",
                "is_authorized": False,
                "is_suspect": False,
                "is_unknown": True,
            }],
            auto_alert=False,
        )

        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0]["alertType"], "unknown_intrusion")
        engine.delete_zone(zone["id"])


if __name__ == "__main__":
    unittest.main()
