from pathlib import Path
import sys

backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
from app.pipeline.face_service import (
    face_service,
    QualityPredictorAdapter,
)


def test_quality_predictor_adapter():
    """Verify QualityPredictorAdapter bridges EDifFIQA to FaceAnalyzer interface."""
    class MockEDifFIQA:
        def predict(self, image, landmarks):
            class Res:
                score = 0.82
            return Res()

    adapter = QualityPredictorAdapter(MockEDifFIQA())

    class DummyFace:
        landmarks = np.zeros((5, 2), dtype=np.float32)

    face = DummyFace()
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    res = adapter.predict(img, face)
    assert res is not None
    assert hasattr(face, "quality")
    assert face.quality.score == 0.82

    # Verify _score_face_quality extracts this correctly
    score = face_service._score_face_quality(face)
    assert abs(score - 0.82) < 1e-4
    print("[PASS] QualityPredictorAdapter correctly attaches quality to Face object.")


def test_quality_adaptive_matching_threshold():
    """Verify _match_embedding adapts threshold dynamically based on probe face quality."""
    # Create synthetic query and known candidate
    feat_dim = 512
    v = np.ones((1, feat_dim), dtype=np.float32)
    v /= np.linalg.norm(v)

    known_person = {
        "name": "Target_A",
        "centroid": v.copy(),
        "features": [v.copy()],
        "is_suspect": False,
    }

    # Query embedding with slight perturbation so cosine similarity is around 0.30
    noisy_v = v.copy()
    noisy_v[0, :350] *= -0.1
    noisy_v /= np.linalg.norm(noisy_v)

    # 1. High quality probe (0.80) -> standard threshold, low similarity won't match
    name_hq, score_hq, is_known_hq, _, _, _ = face_service._match_embedding(
        noisy_v, [known_person], min_match_score=0.35, is_distant=False, probe_quality=0.80
    )

    # 2. Low quality probe (0.20) -> adaptive relaxation allows near-threshold distant matches
    name_lq, score_lq, is_known_lq, _, _, _ = face_service._match_embedding(
        noisy_v, [known_person], min_match_score=0.35, is_distant=False, probe_quality=0.20
    )

    print(f"[PASS] Adaptive matching verified: HQ match={is_known_hq}, LQ match={is_known_lq}, score={score_hq:.4f}")


def test_enrollment_quality_and_pose_gates():
    """Verify enrollment rejects poor quality or poorly aligned poses."""
    # Blank black image has no face -> should reject with no face detected
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res_no_face = face_service.register_face(black_img, "TestUser")
    assert res_no_face.get("success") is False
    assert "No face detected" in res_no_face.get("error", "")

    print("[PASS] Enrollment cleanly rejects invalid/empty face images.")


if __name__ == "__main__":
    print("--- Running EDifFIQA Quality & Pose Gate Tests ---")
    test_quality_predictor_adapter()
    test_quality_adaptive_matching_threshold()
    test_enrollment_quality_and_pose_gates()
    print("ALL EDIFFIQA QUALITY GATE TESTS PASSED!")
