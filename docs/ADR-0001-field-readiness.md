# ADR-0001: Park NIGHTWATCH field-ready (honesty pass before a hardware buy)

**Status:** Accepted
**Date:** 2026-06-19
**Deciders:** QuantumDrizzy

## Context

NIGHTWATCH is being parked until a hardware purchase window (Sept–Dec 2026: Jetson + sensor +
possibly a ~€700–800 GoTo telescope). It must sit untouched in the meantime and, when re-cloned to
build the rig, be *genuinely ready* — not a synthetic demo dressed up with inflated language.

A review (see [CODE-REVIEW.md](CODE-REVIEW.md)) found a solid signal-processing + ML spine (CA-CFAR
CUDA kernel, cited synthetic ToF noise model, a real MobileViT-XT with validated ONNX parity, a
clean DB, a genuinely good kinematic classifier in the dashboard) sitting next to:

- a C++ neural-inference step that was a `rand()` stub fabricating `confidence = 0.85`,
- a grid-wide race on the `v_out` accumulators,
- two build paths that disagreed and one (`Makefile`) that omitted sources, plus a Windows-only
  `<direct.h>` dependency and an OpenCV path hard-wired to an Unreal Engine install,
- "quantum" vocabulary on classical DSP (`estimate_quantum_phase`, "Husimi-Q", "visión cuántica"),
  an out-of-scope acoustic module (LITHOS), theatrical branding, and an external DeepSeek API call
  used only to print a fake "military report,"
- ~106 MB of regenerable synthetic data committed to git.

## Decision

Do the **software honesty pass (Roadmap Phase A) now**, with zero hardware spend, and defer every
hardware-dependent step. Specifically:

- Replace the C++ inference stub with a **real classical CFAR-centroid fallback** (weighted centroid
  of the CFAR mask, honest size-based confidence, `0.0` when nothing survives). The neural path
  activates only when built with `NIGHTWATCH_USE_TENSORRT` + an engine; `initialize()` now fails
  loudly instead of pretending.
- Fix the `v_out` race (host `cudaMemset` per frame; no in-kernel reset).
- Unify the build (both paths compile the same 4 sources), make `main.cpp` portable
  (`std::filesystem` instead of `<direct.h>`), and make the OpenCV location configurable.
- De-inflate: remove "quantum"/"Husimi-Q" naming, theatrics, and the external DeepSeek call; fix the
  chi-square DOF labels; match the dashboard's servo command to the firmware's `$SLEW` parser.
- Remove the out-of-scope LITHOS acoustic module (preserved in git history; can become its own repo).
- Stop tracking `RAW_DATA/` (regenerable); keep model weights, the RAG SQLite store, and the README figure.

## Options Considered

### Option A: Wire real TensorRT inference now
| Dimension | Assessment |
|-----------|------------|
| Complexity | High |
| Cost | TensorRT SDK + an INT8 engine; needs the Jetson to be meaningful |
| Verifiability | Can't honestly verify throughput without the target hardware |

**Pros:** the integrated neural path would be real end-to-end.
**Cons:** unverifiable now; couples the park to a build we can't run yet; high effort for a deferred payoff.

### Option B: Honesty pass + classical fallback, defer neural/hardware (**chosen**)
**Pros:** the default build does something *real and honest* today (a classical detector); every
claim in the repo matches the running code; nothing is unverifiable; cheap and done now.
**Cons:** the neural classifier remains a Python-validated artifact not yet executed in the C++ loop
(documented as a `[KNOWN_LIMIT]`).

### Option C: Leave as-is, fix later
**Pros:** zero effort now.
**Cons:** a public repo (part of the portfolio) keeps inflated/dishonest naming and a fake-confidence
stub; re-cloning in Sept means re-discovering all of this cold.

## Consequences

- **Easier:** a stranger can `git clone` and build the synthetic pipeline on Linux *or* Windows; the
  running binary's behavior matches the docs; re-cloning in Sept gives a clean, coherent base.
- **Harder / deferred:** real neural inference in C++ (needs TensorRT + engine), live sensor capture,
  and any latency/throughput benchmark — all gated on hardware (see [ROADMAP.md](ROADMAP.md) Phase B+).
- **To revisit:** when the Jetson + sensor land, implement the TensorRT engine loader in
  `trackformer_trt.cpp`, wire live capture behind `NIGHTWATCH_USE_SENSOR`, and publish the first
  honest benchmarks.

## `[KNOWN_LIMIT]` (carried, documented honestly)
- The integrated **neural** path is not executed in the default C++ build — it runs the classical
  CFAR-centroid fallback. The MobileViT-XT is validated only in Python (ONNX parity), not yet in the
  compiled pipeline.
- The CUDA/C++ build was **not** compiled/run during this pass (no toolchain invoked here);
  correctness of the kernel/host changes is by inspection. First real build happens on the Jetson.
- ONNX runs at batch=1 only (LSTM initial-state baked at export) — already documented + tested.
