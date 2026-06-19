# NIGHTWATCH — Code Review (2026-06)

Honest review of the repo as cloned from `QuantumDrizzy/NIGHTWATCH` @ `main` (`38bec67`).
Severity: **P0** = blocks honest use / correctness · **P1** = honesty/coherence · **P2** = polish.

> **Resolution (2026-06-19, see [ADR-0001](ADR-0001-field-readiness.md)).** The Phase A honesty
> pass addressed most items below: **P0.1** (stub → real classical CFAR-centroid fallback, neural
> gated on TensorRT), **P0.2** (`v_out` race fixed via host `cudaMemset`), **P0.3** (builds unified,
> `main.cpp` made portable with `std::filesystem`, OpenCV path configurable), **P1.1/P1.3** (quantum
> naming + theatrics + DeepSeek call removed), **P1.2** (LITHOS removed — in git history), **P1.4**
> (`RAW_DATA/` untracked), **P1.5** + part of **P2** (chi-square labels fixed; dashboard servo command
> now matches the `$SLEW` firmware parser). **Deferred to hardware:** real C++ neural inference,
> live capture, benchmarks, the TLE KD-tree, the 512×424/640×480 unification (pick at sensor time).
> The findings below are kept as the original snapshot.

---

## What is real and works (credit where due)

- **CA-CFAR 2D kernel** (`vision_kernel.cu`) — a correct cell-averaging CFAR (training radius 4,
  guard radius 1, empirical `ALPHA=3.5`). The comment honestly says ALPHA is set empirically,
  not derived from a target Pfa. Reads global memory directly (no shared-mem tiling) — correct
  but unoptimized.
- **Synthetic ToF generator** (`synth_tof.cu`) — genuinely good: thermal floor, multipath ripple,
  shot noise, inverse-square falloff, specular saturation, all cited to real literature
  (Sarbolandi 2015, Dal Mutto 2012). Constant-memory scene broadcast, per-pixel LCG + Box-Muller.
- **MobileViT-XT** (`trackformer.py`) — builds, runs, trains, exports. The factorized
  spatiotemporal attention + differential temporal tokens is a reasonable design. ONNX parity is
  **validated and tested** (`tests/test_onnx_parity.py`, max diff ~1e-7 @ batch=1).
- **DB layer** (`nightwatch_db.py`) — clean: configurable path via env, non-destructive column
  migration, parametrized queries. No SQL injection, no surprises.
- **TLE cross-match** (`space_oracle.py`) — works offline against cached Celestrak, 7-day cache
  freshness logic, Skyfield topocentric altaz. Honest inline note that a real system needs a
  KD-tree.
- **Firmware** (`nightwatch_mega.ino`) — reasonable non-blocking loop, NMEA-style checksummed
  telemetry, proportional slew smoothing.
- **README "Status" section** — already honest: "Benchmarks not yet published — measured honestly
  rather than asserted." The `[KNOWN_LIMIT]` on ONNX batch=1 is documented and tested. Keep this tone.

---

## P0 — correctness / blocks honest use

### P0.1 — The C++ neural path is a `rand()` stub
`trackformer_trt.cpp::infer()` with `NIGHTWATCH_USE_TENSORRT` undefined (the default — it's
commented out in `trackformer_trt.h:9`) returns:
```cpp
seed.x = (width/2.0f) + (std::rand() % 10 - 5);
seed.confidence = 0.85f;  // hardcoded
seed.vx = 1.2f; seed.vy = 0.5f;  // constant
```
So the running binary feeds the Kalman gate a random jitter around frame center, **not** the
MobileViT output. `initialize()` never loads the `.engine` either — it just prints and sets a flag.
**The CFAR (real) and the network (validated in Python) are not connected in the default build.**
→ Either wire real TensorRT inference, or relabel the C++ path as a *harness/skeleton* in the
README and make the integrated demo run through Python (PyTorch/ONNXRuntime) where the model
actually executes.

### P0.2 — `v_out` accumulator race in `process_vision_kernel`
`vision_kernel.cu:28-31`: thread (0,0) zeroes `v_out`, then `__syncthreads()`. But `__syncthreads()`
only syncs **within a block**, not the grid — other blocks' `atomicAdd(&v_out->...)` race against
the zeroing. `dx/dy/energy/sharpness` are therefore unreliable across frames.
→ Zero `v_out` from the host (`cudaMemset`) before each `launch_vision_kernel`, remove the
in-kernel reset + `__syncthreads()`.

### P0.3 — Build paths are inconsistent and one is broken
- `Makefile` compiles only `main.cpp vision_kernel.cu` — **omits `synth_tof.cu` and
  `trackformer_trt.cpp`** → link error (`launch_synth_frame`, `TrackformerTRT` undefined). It also
  links `-lfreenect` (Kinect driver) which `build.bat` doesn't, and uses a non-ASCII target name
  `kħaos_vision`.
- `build.bat` hardcodes the OpenCV DLL/include to an **Unreal Engine 5.7 install path**
  (`C:\Program Files\Epic Games\UE_5.7\...`). Nobody else has that; it ties an IR sky-watcher to a
  game engine for one DLL.
→ Make both build paths compile the same 4 sources; get OpenCV from `pkg-config`/`vcpkg`/a
configurable env var, not from UE; document the optional `-lfreenect` Kinect path explicitly.

---

## P1 — honesty / coherence (matches your engineering standard)

### P1.1 — "Quantum" vocabulary on classical DSP
- `acoustic/lithos_qpe.cpp::estimate_quantum_phase()` is a **single-bin DFT** — complex inner
  product of the audio buffer with one reference sinusoid, returning `std::arg`. That's classical
  Goertzel-style phase detection. Calling it "quantum phase estimation" (a specific quantum
  algorithm) is exactly the inflation your own rules forbid.
- `main.cpp` comments: "vision cuantica" — it's temporal IIR accumulation, nothing quantum.
→ Rename to what it is (`estimate_tone_phase`, "temporal accumulation").

### P1.2 — The `acoustic/` (LITHOS) module is out of scope
A 96 kHz microphone phase-tracker has no connection to overhead-object detection and is wired into
nothing. It's scope creep inside a sky-watcher.
→ Either give it a stated role (and connect it) or split it out to its own repo. Don't ship it as
"complementary acoustic-sensing" without a defined function.

### P1.3 — Theatrical / cross-project branding
`[VANGUARDIA 4.1]`, "División de Arquitectura de IA de NIGHTWATCH", `KĦAOS-TRACKFORMER`,
`kħaos_vision`, and a literal comment "simulador ... de Claude" in `main.cpp:100`. Fictional
divisions and tool-name drops read as noise to an engineer reviewer.
→ Strip the theatrics. Keep names descriptive.

### P1.4 — Repo bloat: regenerable data committed
`.git` is **103 MB**; `RAW_DATA/` is **106 MB** of synthetic `.bin` frames that `main
--generate-dataset` regenerates on demand, plus `.pth`/`.onnx`/`.sqlite` artifacts. This violates
the "datasets/build artifacts → gitignore, never pushed" rule and bloats every clone permanently.
→ `.gitignore` `RAW_DATA/`, ship the model weights only if you want zero-retrain UX (or via a
release asset), and consider `git filter-repo` to drop the history bloat. (History rewrite = force
push; only with your explicit OK.)

### P1.5 — "Kalman" overstated + chi-square DOF label wrong
`main.cpp` `kalman_update` uses a scalar covariance `p`, not a matrix; the Mahalanobis gate is a 2D
position distance. `GAMMA_SQ = 9.21f` is labeled "chi-square 4 DOF, 0.99" but 9.21 is the **2-DOF**
0.99 quantile (4-DOF@0.99 = 13.28). The math (2D gate → 2 DOF) is fine; the label is wrong, and
calling it a Kalman filter oversells a 1st-order α-tracker.
→ Fix the comment, or implement the real 2×2 (or 4×4) covariance update and call it a Kalman filter
honestly.

---

## P2 — polish

- **Resolution mismatch**: `nightwatch_vision.h` `#define`s 512×424 (Kinect v2 ToF res) but
  `main.cpp` hardcodes 640×480 and the net wants 64×64. The header constants are dead. Pick one
  source of truth.
- **`space_oracle.find_match`** is O(N) over ~10k active sats per query with a bare `except:`.
  Fine for a demo, won't keep real-time cadence. Add the KD-tree (or pre-filter by orbital plane)
  and narrow the except.
- **`ifu_mode`** kernel param is unused. "sharpness" is Σ(laplacian²) but commented "varianza".
- **`init_db()` runs on import** (side effect) — harmless but surprising; consider explicit init.
- **`omega` / `entropy`** columns/fields exist with no documented meaning.

---

## Bottom line
The signal-processing spine (CFAR + synth ToF) and the Python model (build/train/export/validate)
are real and decent. The gaps are **integration** (the C++ net is a stub) and **honesty/hygiene**
(quantum naming on DSP, out-of-scope acoustic module, theatrics, 100 MB of committed regenerable
data). None are hard to fix; doing so makes this defensible to a domain engineer — which is the bar.
See [ROADMAP.md](ROADMAP.md) for sequencing and [HARDWARE.md](HARDWARE.md) for the field build.
