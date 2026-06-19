# NIGHTWATCH — Hardware Guide

What you actually need to run NIGHTWATCH in the field, from the cheapest honest start to a full
autonomous station. Prices are **approximate EUR, Spain market, 2026** — always check current price;
treat these as orders of magnitude, not quotes.

> **Read [ROADMAP.md](ROADMAP.md) first.** You don't buy everything at once. Phase A needs *nothing*
> (synthetic + public footage). Phase B is the minimum real rig. Phase C adds the GoTo mount + Astrum.
> Phase D is the top tier.

> **Sensor honesty up front:** the code ships a *ToF* noise model (close objects, ~metres). ToF is
> **wrong for the sky** — overhead objects are at optical infinity. For real sky-watching you want a
> **sensitive NIR/visible camera + lens**, not a depth sensor. The ToF/Kinect path is a dev harness only.

---

## 0. You already have (dev, €0)
- **RTX 5060 Ti (16 GB)** desktop — train/export the model, run the full pipeline at dev speed,
  build the TensorRT engine. All Phase A work happens here.
- **114EQ Newtonian (manual, no GoTo)** — usable as a host for a guide/finder camera later, but it
  has **no GoTo motors**, so it can't be slewed by software. Fine for visual; not the automation path.

---

## 1. Compute (the brain in the field)

| Tier | Part | ~€ | Notes |
|------|------|----|-------|
| **Min** | **Jetson Orin Nano (8 GB) Super dev kit** | 270–320 | The target platform. Runs TensorRT INT8, ~67 TOPS in Super mode. Sweet spot for portable/monte. |
| Mid | Jetson Orin NX 16 GB (dev kit / Seeed carrier) | 600–800 | 2–3× the Nano, headroom for bigger nets + capture. |
| Top | Jetson AGX Orin 64 GB | ~2000 | Overkill unless you run many streams / allsky + heavy net. |
| Alt | Mini-PC + your RTX (mains power) | — | If the site has power and you don't need battery, a small x86 box beats a Jetson on raw speed. |

**Start: Orin Nano Super.** It's the platform the INT8 export targets and it runs off a power bank.

---

## 2. Sensor (the eye) — the real decision

| Tier | Part | ~€ | Notes |
|------|------|----|-------|
| **Min** | Raspberry Pi HQ Camera (IMX477, NoIR) **or** USB mono cam + CS-mount lens | 60–100 | No IR-cut filter → NIR sensitive. Wide/fast lens (f/1.4–2.0) for sky-patrol. Cheapest way to see real sky. |
| **Mid** | ZWO ASI120MM-Mini (mono) or ASI462MC | 180–350 | What meteor/satellite watchers actually use. Sensitive, low read-noise, USB, well-supported (INDI). |
| Top | Cooled astro cam (ASI533MM Pro / ASI678) | 500–1500 | Cooling kills thermal noise for long unattended runs. For allsky, pair with a fisheye. |
| 🚫 Avoid | True SWIR/thermal IR cameras | 3000+ | Real shortwave-IR is €€€€ and overkill. The "IR" you want here is NIR, which cheap silicon sensors already see. |

**Start: a NoIR / mono USB cam + a fast wide lens.** Mono + sensitive beats megapixels for faint
moving points. Add the lens to match your field of view (wide = patrol, tele = track).

---

## 3. Mount (pointing)

| Tier | Part | ~€ | Notes |
|------|------|----|-------|
| **Min** | 2× hobby servos + pan/tilt bracket (existing Arduino firmware) | 25–40 | Drives a *camera*, not a telescope. 180° range, low precision. Good enough for slew-to-cue on a light cam. |
| Mid | Star tracker (Sky-Watcher Star Adventurer GTi) | ~500 | Sidereal tracking; can carry a small scope/cam. |
| **Top / GoTo** | Sky-Watcher AZ-GTi (~400) or EQ6-R (~1500) | 400–1500 | **The GoTo path you asked about.** Speaks SynScan; driven by software via **INDI/ASCOM**, *not* the Arduino servos. This is what Phase C targets. |

**Note:** for the GoTo future, NIGHTWATCH talks to the mount through **INDI/ASCOM**, and the Arduino
servo rig stays as the cheap camera-pan/tilt option. Don't try to drive a real telescope with hobby servos.

---

## 4. Position & heading (where am I / which way am I pointing)

| Part | ~€ | Notes |
|------|----|-------|
| GPS NEO-6M (in firmware) | ~10 | Position + time. Already wired. |
| MPU6050 IMU (in firmware) | ~5 | Pitch/roll only — **no magnetometer → no absolute azimuth** (firmware `yaw` is always 0). |
| **Magnetometer QMC5883L / HMC5883L** | ~5 | **Add this** for absolute heading on the servo path. |
| Or: BNO055 9-DOF | ~30 | Fused absolute orientation, saves you the sensor-fusion math. |

A GoTo mount sidesteps heading entirely — it knows its pointing after star-alignment. The magnetometer
matters only for the cheap servo/camera path.

---

## 5. Power (for the monte / field)

| Part | ~€ | Notes |
|------|----|-------|
| USB-C PD power bank (100 W, 25 000 mAh) | 60–90 | Runs Orin Nano + cam for a session. |
| Portable power station (EcoFlow River 2, ~256 Wh) | ~200 | All-night runs incl. a mount + dew heater. |
| Dew heater band + controller | 20–35 | Lens fogs at night; this is not optional in the field. |

---

## 6. Misc (don't forget)
- Sturdy tripod / pier for the cam+mount.
- Red headlamp (preserve night vision).
- Weatherproofing / a Pelican-style case for the Jetson if it lives outside.
- microSD (Jetson boot) + a fast NVMe if you record raw clips.

---

## Buy order (tied to the roadmap)

1. **Phase A — €0.** Dev on the RTX, validate on public sky footage. Buy nothing yet.
2. **Phase B — ~€350–420.** Orin Nano Super (~€300) + NoIR/mono USB cam + fast lens (~€80) + a power
   bank you may already own. First real detections, first honest benchmarks.
3. **Phase B+ — +€10–30.** Magnetometer (+ GPS/IMU if not already on hand) for the servo pan/tilt.
4. **Phase C — +€400+.** A GoTo mount (AZ-GTi) when you want software-driven pointing + the Astrum
   push-to loop. This is the telescope integration.
5. **Phase D — +€500+.** Cooled/allsky cam, power station, enclosure for 24/7 autonomy.

**Minimum to do something real with the sky: ~€350** (Orin Nano + a NoIR camera + lens). Everything
above that is range, autonomy, and the GoTo/Astrum closed loop.
