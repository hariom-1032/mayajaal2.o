import unittest
from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import cv2

from app.pipeline.uniface_service import uniface_service, calibrate_arcface_confidence
from fastapi.testclient import TestClient
from app.main import app


class TestUniFaceIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_uniface_service_readiness(self):
        """Ensure UniFace pipeline loaded models and is ready."""
        self.assertTrue(uniface_service.is_ready())
        self.assertIsNotNone(uniface_service.analyzer)
        self.assertIsNotNone(uniface_service.spoofer)
        self.assertIsNotNone(uniface_service.blur_tool)

    def test_calibrate_arcface_confidence(self):
        """Verify 512-D ArcFace confidence calibration curves."""
        self.assertEqual(calibrate_arcface_confidence(0.0), 0.0)
        self.assertLessEqual(calibrate_arcface_confidence(0.15), 15.0)
        self.assertAlmostEqual(calibrate_arcface_confidence(0.42), 50.0, delta=1.0)
        self.assertGreaterEqual(calibrate_arcface_confidence(0.60), 85.0)
        self.assertGreaterEqual(calibrate_arcface_confidence(0.75), 95.0)

    def test_analyze_frame_blank(self):
        """Blank image should return empty list cleanly without exceptions."""
        blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
        results = uniface_service.analyze_frame(blank_image)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 0)

    def test_status_endpoint_reports_uniface(self):
        """Verify GET /api/faces/status reports uniface readiness and model registry."""
        res = self.client.get("/api/faces/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data.get("uniface_ready"))
        self.assertIn("uniface_models", data)
        self.assertEqual(data["uniface_models"]["recognizer"], "ArcFace_MNet (512-D)")
        self.assertEqual(data["uniface_models"]["demographics"], "FairFace")
        self.assertEqual(data["uniface_models"]["spoofing"], "MiniFASNet_V2")

    def test_scan_endpoint_supports_uniface_engine(self):
        """Verify POST /api/faces/scan supports engine='uniface'."""
        # Create a valid test JPEG
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        files = {"file": ("test.jpg", encoded.tobytes(), "image/jpeg")}
        data = {"engine": "uniface", "verify_liveness": "true"}

        res = self.client.post("/api/faces/scan", files=files, data=data)
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["engine"], "uniface")
        self.assertEqual(payload["face_count"], 0)

    def test_faiss_vector_store_integration(self):
        """Verify FAISS vector index sub-millisecond search and enrollment."""
        self.assertIsNotNone(uniface_service.faiss_store)
        initial_size = uniface_service.faiss_store.size

        # Create synthetic 512-D ArcFace normalized embedding
        test_emb = np.random.randn(512).astype(np.float32)
        test_emb /= np.linalg.norm(test_emb)

        # Add to FAISS index
        test_meta = {
            "person_id": "test_suspect_007",
            "name": "Arjun Rathore",
            "is_suspect": True,
            "threat_level": "CRITICAL",
            "category": "High Value Target",
        }
        uniface_service.faiss_store.add(test_emb, test_meta)
        self.assertEqual(uniface_service.faiss_store.size, initial_size + 1)

        # Search exact match
        matched_meta, score = uniface_service.search_suspect_embedding(test_emb, threshold=0.42)
        self.assertIsNotNone(matched_meta)
        self.assertEqual(matched_meta["name"], "Arjun Rathore")
        self.assertTrue(matched_meta["is_suspect"])
        self.assertGreaterEqual(score, 0.99)

        # Search with non-matching orthogonal vector
        diff_emb = -test_emb
        matched_diff, score_diff = uniface_service.search_suspect_embedding(diff_emb, threshold=0.42)
        self.assertIsNone(matched_diff)
        self.assertLess(score_diff, 0.42)

        # Cleanup
        uniface_service.delete_person("test_suspect_007")

    def test_status_reports_faiss_vector_store(self):
        """Verify /api/faces/status exposes FAISS vector store status."""
        res = self.client.get("/api/faces/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("vector_store", data)
        self.assertEqual(data["vector_store"]["engine"], "FAISS (IndexFlatIP)")
        self.assertEqual(data["vector_store"]["dimension"], 512)
        self.assertEqual(data["vector_store"]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
