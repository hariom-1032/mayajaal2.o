# High-Accuracy Face Recognition Pipeline with UniFace Landmarks

This module solves the low-confidence and false "Unknown" issue by introducing:
1. **Standardized 5-Point Affine Alignment (`aligner.py`):** Automatically rotates, scales, and centers faces to standard 112×112 canonical coordinates before feature extraction, making recognition invariant to head tilts.
2. **Enrollment Quality Gate (`quality_gate.py`):** Rejects blurry (Laplacian variance), tilted (roll angle), off-angle (yaw symmetry), or poorly lit frames during enrollment.
3. **Multi-Sample Centroid Registration (`database.py`):** Averages 5–10 high-quality L2-normalized embeddings per user so the system recognizes faces across slight expression and lighting variations.
4. **Symmetrical Live Inference (`recognizer.py`):** Applies the exact same alignment and calculates Cosine Similarity, calibrated to an intuitive 0–100% confidence scale.

---

## Quickstart

### 1. Install Dependencies
```bash
# CPU mode
pip install -r requirements.txt

# Or for NVIDIA GPU CUDA acceleration:
pip install "uniface[gpu]" opencv-python numpy onnxruntime-gpu
```

### 2. Enroll a New Face
Run the interactive enrollment script. The quality gate will guide you via on-screen HUD:
```bash
python enroll.py --name "Your Name" --samples 8
```
- Look directly at the webcam.
- Slowly tilt or turn your head slightly (±5°) to give the model rich sample variation.
- When 8 valid samples are collected, it automatically computes the centroid and saves to `enrolled_faces.pkl`.

### 3. Run Live Recognition
Start real-time recognition with calibrated confidence percentages:
```bash
python recognize.py --threshold 0.50
```
- Known users will display in **Green** with their name and confidence (e.g. `John Doe (93.5%)`).
- Below threshold faces will display in **Red** with `Unknown (0.32)`.
