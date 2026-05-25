
---

### **2. KINECT-NIR (Actualizado)**

```markdown
# KINECT-NIR

**Real-time infrared overhead object detection and tracking pipeline.**

CUDA-accelerated CA-CFAR detection with spatiotemporal deep learning and multi-sensor fusion.

### Key Results
- Real-time processing at **30 Hz** with Kalman tracking and TLE catalog cross-matching
- MobileViT-XT model exported to **TensorRT INT8**
- 5-class classification using KineticMCO v6.0 (catalogued, uncatalogued, aircraft, debris, anomalous)
- Full hardware bridge to Arduino Mega for servo slew-to-cue on anomalous detections

### What it does
KINECT-NIR is an end-to-end pipeline for real-time detection, tracking, and classification of objects in overhead infrared imagery. It combines classical signal processing (CA-CFAR) with a lightweight spatiotemporal neural network and classical tracking, while maintaining compatibility with real-time hardware deployment.

### Stack
- **Languages**: C++17, CUDA, Python
- **Key Technologies**: OpenCV, PyTorch, ONNX, TensorRT, FastAPI, Skyfield, Arduino
- **Target**: Real-time IR object detection and tracking

### Architecture
- CUDA CA-CFAR kernel for blob detection
- MobileViT-XT with factorized spatiotemporal attention (exported to TensorRT INT8)
- Kalman filter with Mahalanobis gating
- TLE catalog cross-match via Celestrak
- KineticMCO v6.0 classifier (5 classes)
- FastAPI + WebSocket dashboard at 30 Hz
- Arduino hardware bridge for servo control

### Build
```bash
# Windows
build.bat

# Linux / WSL2
make
