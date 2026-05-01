# kinect-nir

Kinect v2 IR pipeline for real-time overhead object detection and classification. CA-CFAR on CUDA, MobileViT-XT spatiotemporal tracking, Kalman gating against Celestrak TLE catalog.

Ingests 640×480 16-bit IR frames. Runs CA-CFAR on GPU to extract blob candidates. Feeds 16-frame clips into a MobileViT-XT network. Applies Kalman filtering with Mahalanobis gating. Cross-matches against live TLE data. Classifies detections via KineticMCO v6.0. Streams results to a FastAPI dashboard. Logs to SQLite.

---

## what exists

| component | status |
|---|---|
| CUDA detection kernel (CA-CFAR + density accumulation) | complete |
| synthetic Kinect v2 noise model | complete |
| MobileViT-XT (3D CNN + FSA attention) | complete |
| training pipeline | complete — smoke-tested, 5 epochs |
| ONNX export + parity validation | complete |
| TensorRT C++ wrapper | stub — conditional compile, simulated fallback |
| Kalman filter (host-side) | complete |
| TLE cross-match via Celestrak | complete |
| KineticMCO v6.0 classifier | complete |
| FastAPI dashboard + WebSocket | complete |
| Arduino hardware bridge (IMU + GPS + servo) | complete |
| real Kinect v2 SDK integration | optional, not compiled by default |
| training data | 5 clips × 16 frames — synthetic only, not production-scale |

---

## architecture

```
Kinect v2 IR (640×480 16-bit)
    │
    ├─ synth_kinect.cu
    │      thermal floor (μ=327, σ=25 ADU) + multipath ripple + shot noise
    │      Sarbolandi et al. 2015 noise model
    │
    ├─ vision_kernel.cu
    │      density matrix temporal blend (α=0.90)
    │      CA-CFAR: 9×9 training window / 3×3 guard / threshold factor 3.5
    │      centroid-weighted velocity via atomicAdd
    │
    ├─ trackformer_trt.h / trackformer_trt.cpp
    │      MobileViT-XT inference via TensorRT (stub — see notes)
    │      input: (B, 3, 16, 64, 64)
    │      channels: density · temporal differential · local std
    │
    ├─ main.cpp
    │      Kalman filter, constant-velocity model
    │      Mahalanobis gating (γ² = 9.21, χ² 4-DOF α=0.01)
    │      writes telemetry.jsonl
    │
    ├─ space_oracle.py
    │      Celestrak active-satellite TLE fetch (7-day cache)
    │      Skyfield ephemeris, observer fixed at Madrid
    │      match tolerance: 5°, minimum elevation: 10°
    │
    └─ nightwatch_dashboard.py
           FastAPI + WebSocket at 30 Hz
           KineticMCO v6.0 over 20-frame history buffer
           polar radar + Leaflet tactical map
           serial bridge → Arduino Mega (servo slew-to-cue on Class X)
```

---

## kinetic mco v6.0

Classifies detections on three features: ω (angular velocity, mean over last 5 frames), ε (linearity residual), σ²_B (brightness variance). Decision tree runs top-to-bottom.

| class | label | condition |
|---|---|---|
| A | catalogued | TLE match ∧ d² < 9.21 |
| B | uncatalogued | no TLE match, nominal kinematics |
| C | aircraft / crosser | ω ∈ [0.30, 4.00] °/s or atmospheric signature |
| D | tumbling debris | σ²_B > 0.04 |
| X | anomalous | ω > 8.00 °/s (hypersonic apparent) or ω < 0.005 °/s (static, persistent) |

Class X triggers a `$SLEW` command to the hardware mount.

---

## model

MobileViT-XT with factorized spatiotemporal attention (FSA). SE-attention blocks throughout. ReLU6 everywhere — no GELU, no Swish. INT8-quantization-compatible by construction. Causal temporal masking for real-time deployment. BiLSTM regression head over the time dimension.

Pre-trained weights included in the repo:

```
nightwatch_mobilevit.pth              PyTorch checkpoint
NIGHTWATCH_MOBILEVIT_XT.onnx          ONNX export (opset 14)
NIGHTWATCH_MOBILEVIT_XT.engine        TensorRT INT8 serialized
```

---

## stack

`C++17 · CUDA · OpenCV · PyTorch · ONNX Runtime · TensorRT · FastAPI · uvicorn · Skyfield · SQLite · Arduino`

---

## build

### windows

Requires Visual Studio 2022 C++, CUDA toolkit, OpenCV.

```bat
build.bat
```

OpenCV path is hardcoded to a UE5.7 / Epic Games install. Edit `build.bat` if your OpenCV is elsewhere.

For real Kinect v2 support, set `KINECTSDK20_DIR` before building. Not set by default.

### linux / wsl2

```sh
make
```

Minimal Makefile. Assumes CUDA and OpenCV on PATH.

---

## training

```sh
# generate synthetic dataset (compiles C++, runs nightwatch_vision.exe --generate-dataset 5)
python dataset_generator.py

# train (5-epoch smoke test by default)
python train.py

# export to ONNX
python export.py

# parity check: PyTorch vs ONNX over 100 random forward passes
python validate_onnx.py
```

TensorRT engine — external step, run after ONNX export:

```sh
trtexec \
  --onnx=NIGHTWATCH_MOBILEVIT_XT.onnx \
  --int8 \
  --saveEngine=NIGHTWATCH_MOBILEVIT_XT.engine
```

---

## dashboard

```sh
python nightwatch_dashboard.py
```

Serves on `http://localhost:8000`. Reads `telemetry.jsonl` via WebSocket at 30 Hz. `nightwatch.sqlite` auto-initializes on first run.

Hardware serial bridge defaults to `COM3` at 115200 baud. Edit `nightwatch_dashboard.py` to match your port.

```sh
# backfill SQLite from telemetry.jsonl (last 500 lines)
python populate_db.py

# post-event 4-panel diagnostic (polar, phase space, ω distribution, Class X heatmap)
python analyze_blackbox.py
```

---

## hardware

Arduino Mega 2560 + MPU6050 (I2C) + NEO-6M GPS (Serial1, 9600 baud) + two servo motors on pins 9 and 10.

Flash `nightwatch_mega/nightwatch_mega.ino`. Required libraries: Wire, Servo, TinyGPS++.

Outbound packets: `$ATT` at 1 Hz (roll, pitch, lat, lon, alt + NMEA checksum).
Inbound command: `$SLEW,<az>,<alt>` — smooth servo interpolation at 15 ms intervals.

---

## notes

- observer coordinates are hardcoded to Madrid (40.4168°N, 3.7038°W) in `space_oracle.py`
- the TensorRT wrapper is a stub — fallback returns `confidence=0.85` for every detection; replace with real TRT inference before deploying
- training data is 5 synthetic clips, 16 frames each — not enough for production accuracy
- `build.bat` OpenCV path targets UE5.7; will need editing on any other setup
- `dataset_generator.py` calls DeepSeek-V3.2 via NVIDIA NIM for a validation report — needs a valid NIM API key

---

## related

- [khaos-core](https://github.com/QuantumDrizzy/khaos-core) — BCI kernel, neurorights enforced at the hardware level
- [cryptotn-gpu](https://github.com/QuantumDrizzy/cryptotn-gpu) — GPU engine for quantum biology
- [quantum-geo-metrology](https://github.com/QuantumDrizzy/quantum-geo-metrology) — geophysical + quantum computing
