# NIGHTWATCH

**Real-time detection, tracking and classification of overhead objects in infrared imagery.**

A CUDA-accelerated vision pipeline that finds faint moving objects in IR/ToF frames,
tracks them, cross-matches them against the satellite catalogue, and flags the ones
that don't belong. Classical signal processing (CA-CFAR) produces detections; a Kalman
tracker and a TLE catalogue match turn raw blobs into identified tracks. A lightweight
spatiotemporal neural net (MobileViT-XT) classifies tracks when built with TensorRT;
otherwise the pipeline runs a real classical CFAR-centroid fallback (see Status).
Detections are persisted to a local store the system learns from over time.

<p align="center">
  <img src="nightwatch_analysis.png" alt="NIGHTWATCH detection/tracking analysis" width="720">
</p>

## Pipeline

```
  IR / ToF frames                CUDA                     classify + track
  ───────────────   ─────────────────────────────   ──────────────────────────
  sensor or          CA-CFAR blob detection           MobileViT-XT (spatio-
  synthetic ToF  ─▶  (constant false-alarm rate,  ─▶  temporal, ONNX/TensorRT) ─┐
  (synth_tof.cu)     vision_kernel.cu)                                          │
                                                                               ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │  Kalman tracking (Mahalanobis gating)  →  TLE catalogue cross-match         │
   │  (space_oracle.py, Skyfield/Celestrak)  →  5-class label:                   │
   │     catalogued · uncatalogued · aircraft · debris · anomalous              │
   └───────────────────────────────┬───────────────────────────────────────────┘
                                    ▼
   SQLite store (learns from detections)  →  FastAPI/WebSocket dashboard
   (nightwatch_db.py)                          (nightwatch_dashboard.py)
                                    ▼
   anomalous track  →  slew-to-cue hardware bridge (Jetson + servo, Arduino)
```

## Components

| File | Role |
|------|------|
| `vision_kernel.cu` | CUDA CA-CFAR blob detection (constant false-alarm-rate thresholding) |
| `synth_tof.cu` | Synthetic ToF/IR frame generator — realistic sensor noise, no hardware needed |
| `main.cpp` / `nightwatch_vision.h` | C++/CUDA pipeline orchestrator |
| `trackformer.py` | MobileViT-XT spatiotemporal classifier (factorized attention) |
| `train.py` / `export.py` / `validate_onnx.py` | Train, export to ONNX, validate PyTorch↔ONNX parity |
| `trackformer_trt.{cpp,h}` | TensorRT inference bridge |
| `space_oracle.py` | TLE catalogue cross-match (Skyfield + Celestrak) |
| `nightwatch_db.py` / `populate_db.py` | SQLite detection store (the system learns from what it sees) |
| `nightwatch_dashboard.py` | FastAPI + WebSocket live dashboard + kinematic classifier (KGL) |
| `acoustic/` | LITHOS — complementary acoustic-sensing module (C++) |
| `nightwatch_mega/` | Arduino Mega firmware for servo slew-to-cue (`$SLEW,az,alt`) |

## Sensor input

The pipeline is sensor-agnostic. The default build runs on a **synthetic ToF/IR
generator** (`synth_tof.cu`) with a realistic noise model, so the whole vision stack
can be developed and tested offline. Live capture is a plug-in point: wire a ToF/IR
sensor's SDK into the build (`NIGHTWATCH_USE_SENSOR`) and feed frames to the same
kernels.

## Status

**Implemented:** CUDA CA-CFAR detection · synthetic ToF generator · MobileViT-XT
detector/regressor (train / export / validate) · classical CFAR-centroid fallback in
the C++ pipeline · Kalman tracking · TLE catalogue match · SQLite store · FastAPI
dashboard with a kinematic classifier (KGL) · acoustic LITHOS module. The ONNX export is validated faithful to
the PyTorch model (max diff ~1e-7 at batch=1) by the test suite (`pytest tests/ -q`,
24 tests).

**In progress / deferred to hardware:** real neural inference in C++ (TensorRT engine
loader — the default build runs the classical fallback) · live-sensor capture
integration · hardware slew-to-cue (Jetson + Arduino servo bridge). Benchmarks
(throughput, classification accuracy) are not yet published — they will be measured and
reported honestly rather than asserted here. See [docs/](docs/) for the roadmap.

**[KNOWN_LIMIT]** the integrated MobileViT-XT runs only when built with
`NIGHTWATCH_USE_TENSORRT` + an engine; the default build uses the classical
CFAR-centroid fallback (a real detector, no neural classification). The net is
validated in Python (ONNX parity), not yet executed in the compiled pipeline.

**[KNOWN_LIMIT]** the ONNX runs at batch=1 (the real-time path, one detection cube
at a time); the declared dynamic batch axis does not work past batch=1 because the
exporter fixes the regression-head LSTM initial state — documented and tested.

## Build / run

```bash
# Synthetic vision pipeline (no sensor required)
# Windows:
build.bat --synthetic
# Linux / WSL2:
make

# Train / export the classifier
python train.py
python export.py
python validate_onnx.py
# TensorRT engine:
trtexec --onnx=NIGHTWATCH_MOBILEVIT_XT.onnx --int8 --saveEngine=NIGHTWATCH_MOBILEVIT_XT.engine

# Dashboard
python nightwatch_dashboard.py
```

Requires: CUDA toolkit + nvcc, an MSVC/GCC C++17 toolchain, OpenCV, and the Python
deps in `requirements.txt`. OpenCV is **not** vendored — point `build.bat` at any
OpenCV install via the `NW_OPENCV_INC` / `NW_OPENCV_LIB` / `NW_OPENCV_DLL` environment
variables (defaults target the OpenCV bundled with Unreal Engine 5.7); on Linux/WSL2
the `Makefile` finds it via `pkg-config opencv4`.

## Stack

`C++17` · `CUDA` · `Python` · `OpenCV` · `PyTorch` · `ONNX` · `TensorRT` · `FastAPI` ·
`Skyfield` · `SQLite`

## Documentation

- [docs/CODE-REVIEW.md](docs/CODE-REVIEW.md) — honest review: what works, what's a stub, what to fix.
- [docs/ROADMAP.md](docs/ROADMAP.md) — phased plan from synthetic demo to GoTo-mount + Astrum field station.
- [docs/HARDWARE.md](docs/HARDWARE.md) — field build, minimum → top tier, with approximate prices.

## License

MIT — see [LICENSE](LICENSE).

---

*Local-first IR situational awareness. Synthetic by default, real-sensor ready.*
