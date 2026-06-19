# NIGHTWATCH — Roadmap

A phased plan from "synthetic-only demo" to "autonomous field station that drives a GoTo mount and
talks to Astrum." Each phase is independently useful and ends on something demonstrable. Hardware
for each tier is in [HARDWARE.md](HARDWARE.md).

> Design principle: **the sensor and the mount are plug-in points, not the architecture.** The
> CFAR → track → catalogue-match → contact-event flow stays the same whether the input is a synth
> frame, a USB camera, or a cooled astro cam, and whether the output drives an Arduino pan/tilt or
> an INDI GoTo mount.

---

## Phase A — Software honesty pass (now, summer, **no hardware**)

Goal: make the running binary do what the README says, and remove inflation/bloat. This is the
"defensible to a domain pro" pass. See P0/P1 in [CODE-REVIEW.md](CODE-REVIEW.md).

- [ ] **Wire real inference** in the C++ path, *or* make the integrated demo run through Python
      (ONNXRuntime/PyTorch) where the model actually executes. Retire the `rand()` stub in
      `trackformer_trt.cpp` or clearly label it a skeleton.
- [ ] **Fix the `v_out` grid race** (zero from host with `cudaMemset`, drop the in-kernel reset).
- [ ] **Unify the build** (Makefile + build.bat compile the same sources; OpenCV from a
      configurable path, not Unreal Engine).
- [ ] **De-inflate**: rename `estimate_quantum_phase` → tone-phase; drop "vision cuantica",
      "VANGUARDIA", fictional divisions, the Claude comment.
- [ ] **Decide LITHOS**: connect the acoustic module to something real, or split it to its own repo.
- [ ] **Hygiene**: gitignore `RAW_DATA/`; consider history cleanup for the 100 MB bloat.
- [ ] **Validate on real footage** (no buy): public night-sky / meteor / satellite-pass video
      (e.g. allsky datasets, recorded ISS passes) → feed frames through CFAR + the net, eyeball
      detections. This is the first time the pipeline sees *real* sky instead of synth.

**Exit:** `git clone && build` works for a stranger; the running demo executes the real network;
nothing in the repo is named beyond what it does.

---

## Phase B — First real sensor (autumn, **minimum hardware**)

Goal: replace synthetic frames with live capture and measure honestly.

> **Sensor reality check.** The synth models a *ToF* sensor with targets at ~1.5 m. ToF maxes out at
> a few metres — **useless for the sky** (objects are at optical infinity). For real sky-watching you
> want a **sensitive NIR/visible camera + lens**, not a ToF depth sensor. Keep the ToF/Kinect path
> only as the offline dev harness it already is.

- [ ] Add a `NIGHTWATCH_USE_SENSOR` capture backend feeding `unsigned short* raw_ir` from a real
      camera (V4L2 / vendor SDK) at the native resolution — and **fix the 512×424 vs 640×480
      mismatch** to match the chosen sensor.
- [ ] Capture a few nights of real frames; retrain/fine-tune the MobileViT on real (not synth) cubes.
- [ ] **Publish the first honest benchmarks**: end-to-end FPS on the dev GPU and on the Jetson,
      detection rate vs a hand-labeled clip. Report baseline ("X fps, CFAR+net, batch=1, Orin Nano,
      INT8"), not a bare number.
- [ ] Move TLE match off the O(N) scan (KD-tree / orbital-plane pre-filter) so it keeps cadence.

**Exit:** the system detects a real satellite/aircraft/meteor crossing live video and logs it to the
SQLite store with a TLE label or "uncatalogued."

---

## Phase C — GoTo mount + Astrum integration (when you have a GoTo scope)

Goal: close the **locate → point → display** loop. This is the piece you asked about.

### C.1 — Drive a real GoTo mount (not the hobby servos)
The Arduino + 2× 180° servos is a **camera pan/tilt**, not a telescope driver. Real GoTo mounts
(Sky-Watcher AZ-GTi, EQ6-R, etc.) use steppers and their own protocol. Don't try to bend the servo
firmware to a telescope.
- [ ] Add an **INDI/ASCOM mount backend**: NIGHTWATCH emits a target az/alt (or RA/dec), the INDI
      driver slews the mount. Keep the existing `$SLEW,az,alt` Arduino path as the *cheap pan/tilt*
      option behind the same internal "MountTarget" interface.
- [ ] **Heading fix**: the MPU6050 has no magnetometer, so firmware `yaw` is always 0 → no absolute
      azimuth. For the servo path add a magnetometer (QMC5883L) or a 9-DOF IMU (BNO055). A GoTo
      mount sidesteps this — it knows its own pointing after star-alignment.
- [ ] **Pixel → sky calibration**: replace the placeholder `az = 180 + (x-w/2)*0.1` in `main.cpp`
      with a real plate-solve / calibration matrix (focal length + sensor pitch + mount pointing).

### C.2 — Astrum bridge (the planetarium becomes the display)
NIGHTWATCH is the **detector/brain**; Astrum (your Android planetarium, field-validated) is the
**display/AR**. This closes Astrum's known `locate → AR push-to` gap.
- [ ] Define a small **contact event** contract (reuse the existing FastAPI/WebSocket dashboard):
      `{ts, az, alt, class, confidence, tle_name|null}`.
- [ ] Astrum subscribes to the WS feed and renders a marker / AR arrow at that az/alt → "point your
      phone here, there's an uncatalogued track."
- [ ] **Share the ephemeris/coordinate code** between NIGHTWATCH (`space_oracle`) and Astrum — per
      the astronomy-pillar plan, the shared ephemeris lib is the spine. One source of truth for
      TLE/altaz, consumed by both.

**Exit:** NIGHTWATCH spots a track → slews the GoTo mount to keep it centered → Astrum shows you
where to look. Detector, mount, and planetarium are one loop.

---

## Phase D — Autonomous station (top tier, long-horizon)

- [ ] Cooled / allsky camera for 24/7 unattended watching; dew control; weatherproof enclosure.
- [ ] The SQLite "learns over time" loop actually used: re-train on accumulated real contacts,
      track recurring vs novel objects.
- [ ] Field power + remote telemetry so it runs on the monte without a babysitter.

**Exit:** a sovereign, local, 24/7 IR situational-awareness node — real hardware, honest benchmarks.

---

## Dependency order (short version)
```
A (honesty pass, no buy)
   └─▶ B (min sensor + Jetson, real frames, first benchmarks)
          └─▶ C (GoTo via INDI/ASCOM + Astrum WS bridge + shared ephemeris)
                 └─▶ D (cooled/allsky, autonomous 24/7)
```
Do **A** with zero spend. **B** is the first hardware and unblocks everything real. **C** is the one
you care about (telescope + Astrum) and only makes sense once **B** gives you real detections to point at.
