import asyncio
import json
import math
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

try:
    import serial_asyncio
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[SERIAL] serial_asyncio not installed — hardware bridge disabled.")

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from space_oracle import SpaceOracle
import nightwatch_db

# ── Serial bridge config ───────────────────────────────────────────────────────
SERIAL_PORT    = "COM3"
SERIAL_BAUDRATE = 115200
serial_writer  = None


async def serial_bridge_loop():
    global serial_writer
    first_attempt = True
    while True:
        try:
            if first_attempt:
                print(f"[SERIAL BRIDGE] Buscando puente hardware en {SERIAL_PORT}...")
                first_attempt = False
            if not HAS_SERIAL:
                await asyncio.sleep(60)
                continue
            reader, writer = await serial_asyncio.open_serial_connection(
                url=SERIAL_PORT, baudrate=SERIAL_BAUDRATE)
            print(f"[SERIAL BRIDGE] ¡Conectado al Arduino en {SERIAL_PORT}!")
            serial_writer = writer
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore').strip()
                if line.startswith("$ATT"):
                    try:
                        parts = line.split('*')[0].split(',')
                        if len(parts) >= 7:
                            lat = float(parts[4])
                            lon = float(parts[5])
                            if lat != 0.0 and lon != 0.0 and hasattr(oracle, 'update_observer'):
                                oracle.update_observer(lat, lon)
                    except Exception as e:
                        print(f"[SERIAL BRIDGE] Error parseando trama ATT: {e}")
        except Exception:
            serial_writer = None
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(serial_bridge_loop())
    yield
    task.cancel()


app    = FastAPI(lifespan=lifespan)
oracle = SpaceOracle()


# ══════════════════════════════════════════════════════════════════════════════
# KINEMATIC GATING LAYER  (KGL) — MCO v6.0
# ──────────────────────────────────────────────────────────────────────────────
# Replaces the stateless uap_violation_count with a kinematic discriminant.
# Three features augment the existing TLE / Mahalanobis pipeline:
#   • ω  — mean angular velocity (°/s) over the last 5 inter-frame intervals
#   • ε  — MSE of a linear fit to Az/Alt history  (low → straight track)
#   • σ² — p_det variance per track  (high → tumbling / debris)
# ══════════════════════════════════════════════════════════════════════════════

class TrackRecord:
    """Per-track kinematic state accumulated across frames."""
    HISTORY = 20   # ~0.66 s at 30 Hz

    def __init__(self):
        self.positions  = deque(maxlen=self.HISTORY)  # (az, alt, unix_ts)
        self.brightness = deque(maxlen=self.HISTORY)  # p_det per frame
        self.age        = 0

    def update(self, az: float, alt: float, p_det: float, ts: float) -> None:
        self.positions.append((az, alt, ts))
        self.brightness.append(p_det)
        self.age += 1

    @property
    def angular_velocity(self) -> float:
        """Mean angular rate (°/s) from the last 5 inter-frame deltas."""
        if len(self.positions) < 2:
            return 0.0
        pts    = list(self.positions)[-6:]
        deltas = []
        for i in range(1, len(pts)):
            daz  = pts[i][0] - pts[i-1][0]
            dalt = pts[i][1] - pts[i-1][1]
            dt   = pts[i][2] - pts[i-1][2]
            if dt > 1e-6:
                deltas.append(math.hypot(daz, dalt) / dt)
        return sum(deltas) / len(deltas) if deltas else 0.0

    @property
    def linearity_residual(self) -> float:
        """MSE (°²) of a linear least-squares fit to Az/Alt history.
        Low  → straight track  (satellite-like).
        High → curved track    (aircraft / UAP manoeuvring)."""
        pts = list(self.positions)
        if len(pts) < 4:
            return 0.0
        n   = len(pts)
        xs  = [p[0] for p in pts]
        ys  = [p[1] for p in pts]
        mx, my = sum(xs) / n, sum(ys) / n
        ssxx = sum((x - mx) ** 2 for x in xs) + 1e-9
        ssxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        b    = ssxy / ssxx
        a    = my - b * mx
        return sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n)) / n

    @property
    def brightness_variance(self) -> float:
        """p_det variance.  High → tumbling object → debris candidate."""
        b = list(self.brightness)
        if len(b) < 3:
            return 0.0
        m = sum(b) / len(b)
        return sum((x - m) ** 2 for x in b) / len(b)


class KineticMCO:
    """
    MCO v6.0 — Kinematic Multi-Class Orbital discriminator.

    Decision tree (evaluated top-to-bottom, first match wins):
      D  →  TLE match   AND d²<γ  AND σ²_B > TUMBLE_THRESH
      A  →  TLE match   AND d²<γ
      C  →  OMEGA_MIN < ω < OMEGA_MAX  AND ε > LIN_THRESH    (aircraft)
      C  →  TLE match   AND d²≥γ                              (atmospheric crosser)
      X  →  d²>γ  AND age≥PERSIST  AND (ω>HYPER OR ω<STATIC)
      B  →  everything else (uncatalogued, kinematics nominal)

    All thresholds are empirical defaults — calibrate with logged field data.
    """

    # Mahalanobis gate: 2D az/alt position → chi-square 2-DOF, 0.99 quantile = 9.21
    GAMMA_SQ            = 9.21

    # Aircraft kinematic window
    OMEGA_AIRCRAFT_MIN  = 0.30    # °/s
    OMEGA_AIRCRAFT_MAX  = 4.00    # °/s
    LINEARITY_AIRCRAFT  = 0.08    # °²  MSE

    # UAP anomaly thresholds
    OMEGA_UAP_HYPER     = 8.00    # °/s  apparent hypersonic acceleration
    OMEGA_UAP_STATIC    = 0.005   # °/s  anomalous hover
    UAP_PDET_MIN        = 0.60    # min detector confidence to flag static UAP
    UAP_PERSIST_FRAMES  = 6       # track must be ≥ this age before Class-X

    # Debris / tumbling
    BRIGHTNESS_TUMBLE   = 0.04    # p_det² variance threshold

    def __init__(self):
        self._records: dict = {}

    def update(self, track_id: str, az: float, alt: float,
               p_det: float, ts: float) -> TrackRecord:
        if track_id not in self._records:
            self._records[track_id] = TrackRecord()
        rec = self._records[track_id]
        rec.update(az, alt, p_det, ts)
        return rec

    def classify(self, track_id: str, tle_name: Optional[str],
                 d2: float, p_det: float):
        """Return (mco_class: str, label: str)."""
        rec = self._records.get(track_id)
        if rec is None or rec.age < 2:
            return "B", "Inicializando KGL..."

        omega = rec.angular_velocity
        lin   = rec.linearity_residual
        bvar  = rec.brightness_variance

        # ── Clase D: TLE match + tumbling brightness signature ────────────
        if tle_name and d2 < self.GAMMA_SQ and bvar > self.BRIGHTNESS_TUMBLE:
            return "D", f"Debris~{tle_name} σ²={bvar:.3f} ω={omega:.2f}°/s"

        # ── Clase A: clean TLE correlation ───────────────────────────────
        if tle_name and d2 < self.GAMMA_SQ:
            return "A", f"{tle_name} ω={omega:.2f}°/s"

        # ── Clase C: aircraft kinematic signature ────────────────────────
        if (self.OMEGA_AIRCRAFT_MIN < omega < self.OMEGA_AIRCRAFT_MAX
                and lin > self.LINEARITY_AIRCRAFT):
            return "C", f"Aircraft ω={omega:.1f}°/s ε={lin:.3f}"

        # ── Clase C: TLE match but d² too high (atmospheric crosser) ────
        if tle_name and d2 >= self.GAMMA_SQ:
            return "C", f"Atm~{tle_name} d²={d2:.1f}"

        # ── Clase X: verified kinematic anomaly, persistent track ────────
        if d2 > self.GAMMA_SQ and rec.age >= self.UAP_PERSIST_FRAMES:
            if omega > self.OMEGA_UAP_HYPER:
                return "X", f"!UAP HYPER ω={omega:.1f}°/s d²={d2:.1f}"
            if omega < self.OMEGA_UAP_STATIC and p_det > self.UAP_PDET_MIN:
                return "X", f"!UAP STATIC ω={omega:.4f}°/s p={p_det:.2f}"

        # ── Clase B: uncatalogued, kinematics nominal ────────────────────
        return "B", f"UCT ω={omega:.2f}°/s d²={d2:.1f}"

    def prune(self, active_ids: set) -> None:
        """Remove stale track records to prevent unbounded memory growth."""
        stale = [k for k in self._records if k not in active_ids]
        for k in stale:
            del self._records[k]


# ══════════════════════════════════════════════════════════════════════════════
# TACTICAL DASHBOARD  —  Glassmorphism Aerospace UI
# ══════════════════════════════════════════════════════════════════════════════

html_content = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>NIGHTWATCH-CORE | KGL v6.0</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:           #060a0e;
      --glass-bg:     rgba(8, 18, 26, 0.82);
      --glass-border: rgba(0, 180, 90, 0.15);
      --glass-dim:    rgba(0, 180, 90, 0.07);
      --text-dim:     #4a7a5a;
      --text-mono:    #72a882;
      --accent:       #00ff88;
      --cls-a:        #00ff88;
      --cls-b:        #ffff00;
      --cls-c:        #00ddff;
      --cls-d:        #ff8800;
      --cls-x:        #ff2222;
    }

    html, body {
      width: 100%; height: 100%;
      background: var(--bg);
      color: var(--text-mono);
      font-family: 'Inter', sans-serif;
      overflow: hidden;
    }

    /* ── TOP BAR ── */
    #topbar {
      position: fixed; top: 0; left: 0; right: 0; height: 46px;
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 18px;
      background: rgba(4, 8, 12, 0.95);
      border-bottom: 1px solid var(--glass-border);
      backdrop-filter: blur(16px);
      z-index: 200;
    }
    .brand {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px; font-weight: 700;
      color: var(--accent); letter-spacing: 0.18em;
    }
    .brand em { color: #2a5a3a; font-style: normal; font-weight: 300; }
    .sys-row {
      display: flex; gap: 22px; align-items: center;
      font-family: 'JetBrains Mono', monospace;
      font-size: 10px; color: var(--text-dim);
    }
    #ws-status { display: flex; align-items: center; gap: 6px; }
    #ws-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--cls-x); box-shadow: 0 0 6px var(--cls-x);
      transition: background .3s, box-shadow .3s;
    }
    #ws-dot.online { background: var(--accent); box-shadow: 0 0 8px var(--accent); }

    /* ── LAYOUT ── */
    #main { position: fixed; top: 46px; bottom: 0; left: 0; right: 0; display: flex; }

    #visual-area {
      flex: 1; min-width: 0;
      display: flex; flex-direction: row;
      gap: 15px; padding: 15px;
    }

    /* ── RADAR ── */
    #radar-area {
      flex: 1; min-width: 0;
      display: flex; align-items: center; justify-content: center;
      background: var(--glass-bg); border: 1px solid var(--glass-border);
      border-radius: 8px; position: relative;
    }
    #radar-wrap { position: relative; }
    #radar {
      display: block; border-radius: 50%;
      border: 1px solid rgba(0, 200, 100, 0.28);
      box-shadow: 0 0 0 1px rgba(0,200,100,.06), 0 0 50px rgba(0,255,136,.06),
                  inset 0 0 60px rgba(0,0,0,.7);
    }
    .az-label {
      position: absolute;
      font-family: 'JetBrains Mono', monospace;
      font-size: 9px; color: rgba(0,180,90,.55);
      pointer-events: none; transform: translate(-50%,-50%);
    }
    .az-label.cardinal { color: rgba(0,220,110,.8); font-weight: 600; }

    /* ── MAP ── */
    #map-area {
      flex: 1; min-width: 0;
      background: var(--glass-bg); border: 1px solid var(--glass-border);
      border-radius: 8px; overflow: hidden; position: relative;
    }
    #map { width: 100%; height: 100%; background: transparent; }

    /* ── SIDEBAR ── */
    #sidebar {
      width: 292px; flex-shrink: 0;
      display: flex; flex-direction: column; gap: 10px;
      padding: 10px 14px 10px 0; overflow: hidden;
    }
    .gpanel {
      background: var(--glass-bg);
      border: 1px solid var(--glass-dim);
      border-radius: 7px; backdrop-filter: blur(14px);
      padding: 11px 13px;
    }
    .ptitle {
      font-family: 'JetBrains Mono', monospace;
      font-size: 8.5px; font-weight: 700;
      letter-spacing: 0.22em; text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 9px; padding-bottom: 6px;
      border-bottom: 1px solid var(--glass-dim);
    }
    .mrow { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; }
    .mlabel { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #3a6a4a; }
    .mval {
      font-family: 'JetBrains Mono', monospace;
      font-size: 14px; font-weight: 600; color: var(--accent);
      transition: color .25s;
    }

    /* Tracks list */
    #tracks-list { display: flex; flex-direction: column; gap: 4px; max-height: 200px; overflow-y: auto; }
    #tracks-list::-webkit-scrollbar { width: 2px; }
    #tracks-list::-webkit-scrollbar-thumb { background: var(--glass-border); border-radius: 2px; }
    .trow {
      display: grid; grid-template-columns: 24px 1fr auto;
      gap: 6px; align-items: center;
      padding: 5px 7px; border-radius: 4px;
      background: rgba(0,0,0,.35); border-left: 2px solid;
      font-family: 'JetBrains Mono', monospace; font-size: 9px;
      cursor: pointer; transition: background .15s;
    }
    .trow:hover    { background: rgba(0,255,136,.05); }
    .trow.selected { background: rgba(0,255,136,.08); }
    .tbadge  { font-size: 8px; font-weight: 700; text-align: center; }
    .tinfo   { color: #5a8a6a; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tcoords { color: #3a5a4a; font-size: 8px; text-align: right; white-space: nowrap; }

    .cls-A { color: var(--cls-a); border-color: var(--cls-a); }
    .cls-B { color: var(--cls-b); border-color: var(--cls-b); }
    .cls-C { color: var(--cls-c); border-color: var(--cls-c); }
    .cls-D { color: var(--cls-d); border-color: var(--cls-d); }
    .cls-X { color: var(--cls-x); border-color: var(--cls-x); animation: pulse-x 1s ease-in-out infinite; }

    @keyframes pulse-x {
      0%,100% { opacity:1;   box-shadow: 0 0 5px  rgba(255,34,34,.5); }
      50%      { opacity:.65; box-shadow: 0 0 14px rgba(255,34,34,1);  }
    }
    @keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0} }

    /* Legend */
    .legend-row { display:flex; align-items:center; gap:7px; margin-bottom:5px;
                  font-family:'JetBrains Mono',monospace; font-size:9px; }
    .ldot  { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
    .lkey  { font-weight:700; }
    .lval  { color:var(--text-dim); }

    /* Log */
    #log-list {
      display:flex; flex-direction:column; gap:2px;
      overflow-y:auto; max-height:160px;
      font-family:'JetBrains Mono',monospace; font-size:8.5px; line-height:1.5;
    }
    #log-list::-webkit-scrollbar { width:2px; }
    #log-list::-webkit-scrollbar-thumb { background:var(--glass-border); border-radius:2px; }
    .lentry        { color:var(--text-dim); }
    .lentry .ts    { color:#253525; }
    .lentry .lcls  { font-weight:700; }
    .lentry.cls-A .lcls { color:var(--cls-a); }
    .lentry.cls-B .lcls { color:var(--cls-b); }
    .lentry.cls-C .lcls { color:var(--cls-c); }
    .lentry.cls-D .lcls { color:var(--cls-d); }
    .lentry.cls-X .lcls { color:var(--cls-x); }

    @media (max-width: 640px) { #sidebar { display:none; } }
  </style>
</head>
<body>

  <header id="topbar">
    <div class="brand">NIGHTWATCH<em>-CORE</em> &nbsp;|&nbsp; KGL v6.0</div>
    <div class="sys-row">
      <span id="frm-ctr">FRM 000000</span>
      <span id="trk-ctr">TRK 0</span>
      <div id="ws-status">
        <div id="ws-dot"></div>
        <span id="ws-label">OFFLINE</span>
      </div>
    </div>
  </header>

  <div id="main">
    <div id="visual-area">

      <!-- Radar polar -->
      <div id="radar-area">
        <div id="radar-wrap">
          <canvas id="radar"></canvas>
        </div>
      </div>

      <!-- Tactical map -->
      <div id="map-area">
        <div id="map"></div>
      </div>

    </div>

    <aside id="sidebar">

      <!-- KGL Metrics -->
      <div class="gpanel">
        <div class="ptitle">KGL Métricas · Contacto Activo</div>
        <div class="mrow"><span class="mlabel">MCO Clase</span><span class="mval" id="m-cls">—</span></div>
        <div class="mrow"><span class="mlabel">Az / Alt</span><span class="mval" id="m-pos" style="font-size:12px">— / —</span></div>
        <div class="mrow"><span class="mlabel">D² Mahalanobis</span><span class="mval" id="m-d2">—</span></div>
        <div class="mrow"><span class="mlabel">P_det</span><span class="mval" id="m-pdet">—</span></div>
        <div class="mrow"><span class="mlabel">ω Angular</span><span class="mval" id="m-omega" style="font-size:11px">—</span></div>
      </div>

      <!-- Active contacts -->
      <div class="gpanel">
        <div class="ptitle">Contactos Activos</div>
        <div id="tracks-list">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#253525;">Sin contactos</div>
        </div>
      </div>

      <!-- MCO Legend -->
      <div class="gpanel">
        <div class="ptitle">Leyenda MCO v6</div>
        <div class="legend-row">
          <div class="ldot" style="background:var(--cls-a);box-shadow:0 0 5px var(--cls-a)"></div>
          <span class="lkey" style="color:var(--cls-a)">A</span>
          <span class="lval">Catalogado TLE &nbsp;d²&lt;9.21</span>
        </div>
        <div class="legend-row">
          <div class="ldot" style="background:var(--cls-b)"></div>
          <span class="lkey" style="color:var(--cls-b)">B</span>
          <span class="lval">No catalogado, cinemática nominal</span>
        </div>
        <div class="legend-row">
          <div class="ldot" style="background:var(--cls-c)"></div>
          <span class="lkey" style="color:var(--cls-c)">C</span>
          <span class="lval">Aeronave / Atmosférico</span>
        </div>
        <div class="legend-row">
          <div class="ldot" style="background:var(--cls-d)"></div>
          <span class="lkey" style="color:var(--cls-d)">D</span>
          <span class="lval">Basura orbital (tumbling)</span>
        </div>
        <div class="legend-row">
          <div class="ldot" style="background:var(--cls-x);animation:pulse-dot 1s infinite"></div>
          <span class="lkey" style="color:var(--cls-x)">X</span>
          <span class="lval">Anomalía UAP — KGL confirmado</span>
        </div>
      </div>

      <!-- Event log -->
      <div class="gpanel" style="flex:1;min-height:0;display:flex;flex-direction:column;">
        <div class="ptitle">Registro de Eventos</div>
        <div id="log-list"></div>
      </div>

    </aside>
  </div>

  <script>
    // ── Constants ──────────────────────────────────────────────────────────
    const CLS_COLOR  = { A:'#00ff88', B:'#ffff00', C:'#00ddff', D:'#ff8800', X:'#ff2222' };
    const TRAIL_LEN  = 28;
    const SWEEP_RPM  = 2.0;
    const TRAIL_CONE = Math.PI / 5;   // 36° phosphor cone

    // ── State ──────────────────────────────────────────────────────────────
    let tracks    = {};
    let trackHist = {};
    let msgCount  = 0;
    let selectedId = null;

    // ── Canvas ─────────────────────────────────────────────────────────────
    const canvas = document.getElementById('radar');
    const ctx    = canvas.getContext('2d');
    const wrap   = document.getElementById('radar-wrap');

    // ── Leaflet tactical map ───────────────────────────────────────────────
    const OBS_LAT = 40.4168, OBS_LON = -3.7038;
    const map = L.map('map', { zoomControl: false, attributionControl: false })
                 .setView([OBS_LAT, OBS_LON], 9);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png').addTo(map);

    const obsMarker = L.circleMarker([OBS_LAT, OBS_LON],
      { radius:6, color:'#00ff88', fillColor:'#00ff88', fillOpacity:1 }).addTo(map);
    obsMarker.bindTooltip("NIGHTWATCH BASE",
      { permanent:true, direction:"top", className:"az-label" }).openTooltip();

    const tracksLayer = L.layerGroup().addTo(map);

    function getDestination(lat, lon, brng, distKm) {
      const R  = 6371;
      brng = brng * Math.PI / 180;
      lat  = lat  * Math.PI / 180;
      lon  = lon  * Math.PI / 180;
      const lat2 = Math.asin(
        Math.sin(lat) * Math.cos(distKm/R) +
        Math.cos(lat) * Math.sin(distKm/R) * Math.cos(brng));
      const lon2 = lon + Math.atan2(
        Math.sin(brng) * Math.sin(distKm/R) * Math.cos(lat),
        Math.cos(distKm/R) - Math.sin(lat) * Math.sin(lat2));
      return [lat2 * 180 / Math.PI, lon2 * 180 / Math.PI];
    }

    // ── Canvas resize ──────────────────────────────────────────────────────
    function resizeCanvas() {
      const area = document.getElementById('radar-area');
      const s    = Math.max(260, Math.min(area.clientWidth - 60, area.clientHeight - 36));
      canvas.width  = s;
      canvas.height = s;
      buildAzLabels(s);
    }

    // ── Az/Alt → canvas pixel ──────────────────────────────────────────────
    function toXY(az, alt) {
      const R = canvas.width / 2;
      const r = R * (1 - alt / 90.0);
      const t = (az - 90) * Math.PI / 180;
      return { x: R + r * Math.cos(t), y: R + r * Math.sin(t) };
    }

    // ── Azimuth labels ─────────────────────────────────────────────────────
    const AZ_LABELS = [
      {az:0,  txt:'N', card:true}, {az:45,  txt:'45'},
      {az:90, txt:'E', card:true}, {az:135, txt:'135'},
      {az:180,txt:'S', card:true}, {az:225, txt:'225'},
      {az:270,txt:'W', card:true}, {az:315, txt:'315'},
    ];
    function buildAzLabels(s) {
      wrap.querySelectorAll('.az-label').forEach(el => el.remove());
      const R = s / 2, off = 20;
      AZ_LABELS.forEach(({ az, txt, card }) => {
        const t  = (az - 90) * Math.PI / 180;
        const el = document.createElement('div');
        el.className  = 'az-label' + (card ? ' cardinal' : '');
        el.textContent = txt;
        el.style.left  = (R + (R + off) * Math.cos(t)) + 'px';
        el.style.top   = (R + (R + off) * Math.sin(t)) + 'px';
        wrap.appendChild(el);
      });
    }

    // ── Main render loop ───────────────────────────────────────────────────
    function drawRadar() {
      const W = canvas.width, H = canvas.height;
      const cx = W / 2, cy = H / 2, R = W / 2;

      // Phosphor decay
      ctx.fillStyle = 'rgba(6, 10, 14, 0.20)';
      ctx.fillRect(0, 0, W, H);

      // Polar grid — elevation rings
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 70, 35, 0.55)';
      ctx.lineWidth   = 0.5;
      [0.333, 0.666, 1.0].forEach(f => {
        ctx.beginPath(); ctx.arc(cx, cy, R * f, 0, Math.PI * 2); ctx.stroke();
      });
      // Azimuth spokes
      ctx.strokeStyle = 'rgba(0, 55, 28, 0.5)';
      for (let a = 0; a < 360; a += 45) {
        const t = (a - 90) * Math.PI / 180;
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.lineTo(cx + R * Math.cos(t), cy + R * Math.sin(t)); ctx.stroke();
      }
      // Elevation labels
      ctx.font      = '8px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(0, 110, 55, 0.45)';
      ctx.textAlign = 'left';
      [{f:0.333, alt:60}, {f:0.666, alt:30}].forEach(({f, alt}) => {
        ctx.fillText(alt + '°', cx + R * f + 3, cy - 3);
      });
      ctx.restore();

      // Sweep arm + phosphor cone trail
      const t        = performance.now() / 1000;
      const sweepAng = (t * SWEEP_RPM) % (Math.PI * 2);
      for (let i = 0; i < 24; i++) {
        const ang   = sweepAng - (i / 24) * TRAIL_CONE;
        const alpha = (1 - i / 24) * 0.45;
        ctx.save();
        ctx.strokeStyle = `rgba(0, 255, 136, ${alpha})`;
        ctx.lineWidth   = 1;
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.lineTo(cx + R * Math.cos(ang), cy + R * Math.sin(ang));
        ctx.stroke(); ctx.restore();
      }
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.92)';
      ctx.lineWidth   = 1.5;
      ctx.shadowColor = '#00ff88'; ctx.shadowBlur = 7;
      ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R * Math.cos(sweepAng), cy + R * Math.sin(sweepAng));
      ctx.stroke(); ctx.restore();

      // Center pip
      ctx.fillStyle = 'rgba(0, 255, 136, 0.5)';
      ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fill();

      // Track trails
      Object.keys(trackHist).forEach(id => {
        const hist = trackHist[id];
        if (hist.length < 2) return;
        const col = CLS_COLOR[tracks[id] ? tracks[id].cls : 'B'] || '#fff';
        for (let i = 0; i < hist.length - 1; i++) {
          const alpha = (i / hist.length) * 0.55;
          const p1    = toXY(hist[i].az, hist[i].alt);
          const p2    = toXY(hist[i+1].az, hist[i+1].alt);
          ctx.save();
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = col;
          ctx.lineWidth   = 1;
          ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y);
          ctx.stroke(); ctx.restore();
        }
      });

      // Track blips
      Object.values(tracks).forEach(trk => {
        const {x, y} = toXY(trk.az, trk.alt);
        const col    = CLS_COLOR[trk.cls] || '#fff';
        ctx.save();
        if (trk.cls === 'X') {
          const pulse = 0.5 + 0.5 * Math.sin(t * 6.0);
          ctx.strokeStyle = col; ctx.lineWidth = 1.5;
          ctx.globalAlpha = pulse;
          ctx.shadowColor = col; ctx.shadowBlur = 16;
          ctx.beginPath(); ctx.arc(x, y, 9 + pulse * 4, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1; ctx.shadowBlur = 0;
        }
        ctx.fillStyle   = col;
        ctx.shadowColor = col; ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(x, y, trk.cls === 'X' ? 5 : 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur  = 0; ctx.globalAlpha = 0.82;
        ctx.font        = '8.5px JetBrains Mono, monospace';
        ctx.fillStyle   = col; ctx.textAlign = 'left';
        ctx.fillText((trk.label || '').substring(0, 22), x + 7, y + 3);
        ctx.restore();
      });

      requestAnimationFrame(drawRadar);
    }

    // ── Sidebar ────────────────────────────────────────────────────────────
    function selectTrack(id) { selectedId = id; updateSidebar(); }

    function updateSidebar() {
      const arr  = Object.values(tracks);
      document.getElementById('trk-ctr').textContent = 'TRK ' + arr.length;
      document.getElementById('frm-ctr').textContent = 'FRM ' + String(msgCount).padStart(6, '0');

      const list = document.getElementById('tracks-list');
      if (arr.length === 0) {
        list.innerHTML = '<div style="font-family:JetBrains Mono,monospace;font-size:9px;color:#253525;">Sin contactos</div>';
        ['m-cls','m-pos','m-d2','m-pdet','m-omega'].forEach(id => {
          document.getElementById(id).textContent = '—';
        });
        document.getElementById('m-cls').style.color = '';
        tracksLayer.clearLayers();
        return;
      }

      list.innerHTML = '';
      tracksLayer.clearLayers();

      const clsDistKm = { A:400, B:350, C:30, D:400, X:100 };
      arr.forEach(trk => {
        const distKm = clsDistKm[trk.cls] || 30;
        const dest   = getDestination(OBS_LAT, OBS_LON, trk.az, distKm);
        const color  = CLS_COLOR[trk.cls] || CLS_COLOR['B'];

        L.polyline([[OBS_LAT, OBS_LON], dest],
          { color, weight:1.5, dashArray:'4,6' }).addTo(tracksLayer);
        L.circleMarker(dest,
          { radius: trk.cls === 'X' ? 5 : 3, color, fillColor:color, fillOpacity:0.8 })
         .addTo(tracksLayer)
         .bindTooltip(`[${trk.cls}]`,
           { permanent:true, direction:"bottom", className:"az-label", opacity:0.7 })
         .openTooltip();

        const div = document.createElement('div');
        div.className = 'trow cls-' + trk.cls + (trk.id === selectedId ? ' selected' : '');
        div.innerHTML =
          '<span class="tbadge">' + trk.cls + '</span>' +
          '<span class="tinfo">'  + (trk.label || '').substring(0, 24) + '</span>' +
          '<span class="tcoords">' + trk.az.toFixed(0) + '°/' + trk.alt.toFixed(0) + '°</span>';
        div.onclick = () => { selectedId = trk.id; updateSidebar(); };
        list.appendChild(div);
      });

      const sel = (selectedId && tracks[selectedId]) ? tracks[selectedId] : arr[0];
      if (sel) {
        const mCls = document.getElementById('m-cls');
        mCls.textContent = sel.cls;
        mCls.style.color = CLS_COLOR[sel.cls] || '#fff';
        document.getElementById('m-pos').textContent   = sel.az.toFixed(1) + '°  /  ' + sel.alt.toFixed(1) + '°';
        document.getElementById('m-d2').textContent    = sel.d2    !== undefined ? sel.d2.toFixed(2)             : '—';
        document.getElementById('m-pdet').textContent  = sel.p_det !== undefined ? (sel.p_det*100).toFixed(1)+'%' : '—';
        document.getElementById('m-omega').textContent = sel.omega  !== undefined ? sel.omega.toFixed(3)+'°/s'   : '—';
      }
    }

    // ── Log ────────────────────────────────────────────────────────────────
    const logList = document.getElementById('log-list');
    function addLogEntry(trk) {
      const ts  = new Date().toISOString().substring(11, 22);
      const div = document.createElement('div');
      div.className = 'lentry cls-' + trk.cls;
      div.innerHTML =
        '<span class="ts">'  + ts         + '</span>  ' +
        '<span class="lcls">[' + trk.cls  + ']</span>  ' +
        (trk.label || '').substring(0, 26) +
        '  Az:' + trk.az.toFixed(0) + '°';
      logList.prepend(div);
      if (logList.children.length > 60) logList.removeChild(logList.lastChild);
    }

    // ── WebSocket ──────────────────────────────────────────────────────────
    const ws = new WebSocket("ws://" + location.host + "/ws");

    ws.onopen  = () => {
      document.getElementById('ws-dot').classList.add('online');
      document.getElementById('ws-label').textContent = 'ONLINE';
    };
    ws.onclose = () => {
      document.getElementById('ws-dot').classList.remove('online');
      document.getElementById('ws-label').textContent = 'OFFLINE';
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      tracks = {};
      data.tracks.forEach(t => {
        tracks[t.id] = t;

        if (!trackHist[t.id]) trackHist[t.id] = [];
        trackHist[t.id].push({ az: t.az, alt: t.alt });
        if (trackHist[t.id].length > TRAIL_LEN) trackHist[t.id].shift();

        addLogEntry(t);
        msgCount++;
      });

      // Prune dead trail history
      Object.keys(trackHist).forEach(id => {
        if (!tracks[id]) {
          trackHist[id].shift();
          if (trackHist[id].length === 0) delete trackHist[id];
        }
      });

      updateSidebar();
    };

    // ── Init ───────────────────────────────────────────────────────────────
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();
    requestAnimationFrame(drawRadar);
  </script>
</body>
</html>
"""


# ── HTTP + WebSocket routes ────────────────────────────────────────────────────

@app.get("/")
async def get():
    return HTMLResponse(html_content)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    telemetry_file_path = "telemetry.jsonl"

    # Wait for the C++ engine to create the IPC file
    while not os.path.exists(telemetry_file_path):
        await asyncio.sleep(1)

    # Per-connection KineticMCO instance.
    # Using a PERSISTENT track ID ("trk_0") so the history buffer accumulates
    # correctly across frames and the frontend trail renders properly.
    mco      = KineticMCO()
    TRACK_ID = "trk_0"

    try:
        with open(telemetry_file_path, "r") as f:
            # Seek to end: show only live data, skip backlog
            f.seek(0, os.SEEK_END)

            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.033)   # 30 Hz poll
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                az    = float(data.get("az",    0.0))
                alt   = float(data.get("alt",   0.0))
                d2    = float(data.get("d2",    0.0))
                p_det = float(data.get("p_det", 0.0))
                frame = int  (data.get("frame", 0))

                # ── Space Oracle: TLE correlation ──────────────────────────
                tle_name, _tle_dist = oracle.find_match(az, alt, tolerance_deg=5.0)

                # ── KGL: kinematic classification ──────────────────────────
                ts  = time.time()
                rec = mco.update(TRACK_ID, az, alt, p_det, ts)
                cls, label = mco.classify(TRACK_ID, tle_name, d2, p_det)
                omega = rec.angular_velocity

                # ── Blackbox: persist to SQLite ────────────────────────────
                nightwatch_db.log_contact(
                    mco_class     = cls,
                    azimuth       = az,
                    altitude      = alt,
                    mahalanobis_d2= d2,
                    confidence    = p_det,
                    label         = label,
                    omega         = omega,
                )

                # ── Servo slew-to-cue on Class X ──────────────────────────
                if cls == "X" and serial_writer is not None:
                    try:
                        # Match the firmware's parser: "$SLEW,az,alt\n"
                        # (nightwatch_mega.ino::parseCommand). A JSON command
                        # would be silently ignored by the Arduino.
                        cmd = f"$SLEW,{az:.1f},{alt:.1f}\n"
                        serial_writer.write(cmd.encode())
                    except Exception:
                        pass

                # ── Push to dashboard ──────────────────────────────────────
                track = {
                    "id":    TRACK_ID,
                    "az":    az,
                    "alt":   alt,
                    "cls":   cls,
                    "label": label,
                    "d2":    round(d2,    4),
                    "p_det": round(p_det, 4),
                    "omega": round(omega, 4),
                    "frame": frame,
                }
                await websocket.send_json({"tracks": [track]})
                await asyncio.sleep(0.033)

    except Exception as e:
        print(f"[WS] Client disconnected: {e}")
    finally:
        mco.prune(set())   # release all track memory on disconnect


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
