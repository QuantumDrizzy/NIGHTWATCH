"""
analyze_blackbox.py — NIGHTWATCH Post-Event Analysis
══════════════════════════════════════════════════════════════════════════════
Reads nightwatch.sqlite and renders four diagnostic panels:

  Panel 1 (top-left)  — Polar Az/Alt scatter  (all MCO classes)
  Panel 2 (top-right) — Phase space d² × p_det  (gate visualisation)
  Panel 3 (bot-left)  — ω angular velocity distribution per class (KGL)
  Panel 4 (bot-right) — Class X Az/Alt heat map  (UAP clustering)

Output: nightwatch_analysis.png (150 dpi, dark background)
Usage:  python analyze_blackbox.py [--db nightwatch.sqlite] [--limit 20000]
"""

import argparse
import math
import sqlite3
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                      # headless-safe backend
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ── Palette (mirrors JS CLS_COLOR) ────────────────────────────────────────────
BG      = "#060a0e"
CLS_COL = {
    "A": "#00ff88",
    "B": "#ffff00",
    "C": "#00ddff",
    "D": "#ff8800",
    "X": "#ff2222",
}
CLS_LABEL = {
    "A": "Clase A — Catalogado TLE",
    "B": "Clase B — No catalogado",
    "C": "Clase C — Atmosférico / Aeronave",
    "D": "Clase D — Basura orbital",
    "X": "Clase X — Anomalía UAP",
}

# ── DB helpers ─────────────────────────────────────────────────────────────────

def load_db(db_path: str, limit: int):
    """Return all contact rows as a list of dicts."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT timestamp, mco_class, azimuth, altitude, "
        "mahalanobis_d2, confidence, label, omega "
        "FROM contacts_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def by_class(rows: list):
    """Partition rows into a dict keyed by mco_class."""
    out = {c: [] for c in "ABCDX"}
    for r in rows:
        cls = r.get("mco_class", "B")
        if cls in out:
            out[cls].append(r)
    return out


def ts_to_hour(ts_str: str) -> float:
    """Parse an ISO timestamp string and return fractional UTC hour."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.hour + dt.minute / 60 + dt.second / 3600
    except Exception:
        return 0.0


# ── Axes style helper ──────────────────────────────────────────────────────────

def style_ax(ax, title: str):
    ax.set_facecolor(BG)
    ax.set_title(title, color="#72a882", fontsize=9, pad=8,
                 fontfamily="monospace")
    ax.tick_params(colors="#3a5a4a", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a3a2a")
    ax.xaxis.label.set_color("#4a7a5a")
    ax.yaxis.label.set_color("#4a7a5a")
    ax.xaxis.label.set_fontsize(8)
    ax.yaxis.label.set_fontsize(8)


# ── Panel 1 — Polar scatter ────────────────────────────────────────────────────

def draw_polar_scatter(ax, partitioned: dict):
    """All contacts in Az/Alt polar projection (zenith = centre)."""
    ax.set_facecolor(BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 90)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(["70°", "50°", "30°", "10°"], fontsize=6,
                       color="#3a5a4a")
    ax.tick_params(colors="#3a5a4a", labelsize=7)
    ax.grid(color="#0a2a1a", linewidth=0.4)
    ax.set_facecolor(BG)
    ax.spines["polar"].set_edgecolor("#1a3a2a")

    for cls, rows in partitioned.items():
        if not rows:
            continue
        azs  = [math.radians(r["azimuth"]) for r in rows]
        alts = [90.0 - r["altitude"]       for r in rows]   # invert: zenith=0
        ax.scatter(azs, alts, c=CLS_COL[cls], s=6, alpha=0.55,
                   label=f"{cls}  n={len(rows):,}", zorder=5, linewidths=0)

    ax.legend(loc="lower right", fontsize=6.5,
              facecolor="#0a1a0e", edgecolor="#1a3a2a", labelcolor="white")
    ax.set_title("Distribución Az/Alt — Todos los contactos",
                 color="#72a882", fontsize=9, pad=10, fontfamily="monospace")


# ── Panel 2 — Phase space d² × p_det ──────────────────────────────────────────

def draw_phase_space(ax, partitioned: dict):
    """MCO phase space: Mahalanobis d² (y) vs detector confidence p_det (x)."""
    style_ax(ax, "Espacio de Fase MCO  (d² × p_det)")
    ax.set_xlabel("p_det  (confianza detectora)")
    ax.set_ylabel("d²  (Distancia de Mahalanobis)")

    # Gate reference lines
    ax.axhline(y=9.21, color="#3a1a1a", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(x=0.30, color="#1a3a2a", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(0.31, 9.5, "Puerta χ²=9.21", color="#5a2a2a",
            fontsize=6.5, fontfamily="monospace")
    ax.text(0.01, 0.5, "p_det=0.30", color="#1a4a2a",
            fontsize=6.5, fontfamily="monospace", rotation=90)

    for cls, rows in partitioned.items():
        pts = [(r["confidence"] or 0, r["mahalanobis_d2"] or 0)
               for r in rows if r["confidence"] is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=CLS_COL[cls], s=7, alpha=0.5, zorder=5,
                   label=f"{cls}", linewidths=0)

    ax.legend(fontsize=7, facecolor="#0a1a0e",
              edgecolor="#1a3a2a", labelcolor="white")


# ── Panel 3 — ω distribution (KGL) ────────────────────────────────────────────

def draw_omega_distribution(ax, partitioned: dict):
    """KGL angular velocity ω per MCO class — violin / histogram overlay."""
    style_ax(ax, "ω Angular  (KGL — por clase MCO)")
    ax.set_xlabel("ω  (°/s)")
    ax.set_ylabel("Densidad")

    bins = np.linspace(0, 15, 60)
    for cls, rows in partitioned.items():
        omegas = [r["omega"] for r in rows
                  if r.get("omega") is not None and r["omega"] >= 0]
        if len(omegas) < 3:
            continue
        ax.hist(omegas, bins=bins, density=True, alpha=0.35,
                color=CLS_COL[cls], label=f"Cls {cls}", histtype="stepfilled",
                linewidth=0)
        ax.hist(omegas, bins=bins, density=True, alpha=0.85,
                color=CLS_COL[cls], histtype="step", linewidth=0.8)

    # UAP gate markers
    ax.axvline(x=8.0,  color="#ff2222", linewidth=0.8,
               linestyle=":", label="Hyper (8°/s)")
    ax.axvline(x=0.30, color="#00ddff", linewidth=0.8,
               linestyle=":", label="Aircraft min (0.3°/s)")

    ax.legend(fontsize=6.5, facecolor="#0a1a0e",
              edgecolor="#1a3a2a", labelcolor="white")
    ax.set_xlim(0, 12)


# ── Panel 4 — Class X heat map ─────────────────────────────────────────────────

def draw_x_heatmap(ax, x_rows: list):
    """Az/Alt 2-D density map for Class X events only."""
    style_ax(ax, "Heat Map — Clase X (UAP)")
    ax.set_xlabel("Azimut (°)")
    ax.set_ylabel("Altitud (°)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0,  90)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(45))

    if len(x_rows) < 3:
        ax.text(0.5, 0.5,
                "Datos insuficientes\n(< 3 eventos Clase X)",
                ha="center", va="center", color="#ff2222",
                fontsize=10, transform=ax.transAxes,
                fontfamily="monospace")
        return

    azs  = [r["azimuth"]  for r in x_rows]
    alts = [r["altitude"] for r in x_rows]

    cmap = LinearSegmentedColormap.from_list(
        "nw_x", [BG, "#1a0505", "#5a0000", "#ff2222"], N=256)
    h = ax.hist2d(azs, alts, bins=(36, 18), range=[[0, 360], [0, 90]],
                  cmap=cmap, density=True)
    cb = plt.colorbar(h[3], ax=ax, pad=0.02)
    cb.set_label("Densidad", color="#4a7a5a", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#3a5a4a", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#3a5a4a")

    # Mark centroid of hottest bin
    counts, xedges, yedges = h[0], h[1], h[2]
    max_idx = np.unravel_index(np.argmax(counts), counts.shape)
    hot_az  = (xedges[max_idx[0]] + xedges[max_idx[0]+1]) / 2
    hot_alt = (yedges[max_idx[1]] + yedges[max_idx[1]+1]) / 2
    ax.plot(hot_az, hot_alt, marker="+", markersize=12,
            color="#ff2222", linewidth=1.5, zorder=10)
    ax.text(hot_az + 4, hot_alt + 2,
            f"Az={hot_az:.0f}° Alt={hot_alt:.0f}°",
            color="#ff8888", fontsize=7, fontfamily="monospace")


# ── Summary table (console) ────────────────────────────────────────────────────

def print_summary(rows: list, partitioned: dict):
    total = len(rows)
    print(f"\n{'═'*56}")
    print(f"  NIGHTWATCH Blackbox — {total:,} contactos cargados")
    print(f"{'═'*56}")
    print(f"  {'Clase':<8} {'N':>7}  {'%':>6}  {'ω̄ (°/s)':>10}  {'d̄²':>8}")
    print(f"  {'─'*50}")
    for cls in "ABCDX":
        r = partitioned[cls]
        n = len(r)
        if n == 0:
            continue
        pct    = 100 * n / total if total else 0
        omegas = [x["omega"] for x in r if x.get("omega") is not None]
        d2s    = [x["mahalanobis_d2"] for x in r if x.get("mahalanobis_d2") is not None]
        omega_mean = sum(omegas) / len(omegas) if omegas else float("nan")
        d2_mean    = sum(d2s)    / len(d2s)    if d2s    else float("nan")
        print(f"  {cls:<8} {n:>7,}  {pct:>5.1f}%  {omega_mean:>10.3f}  {d2_mean:>8.2f}")
    print(f"{'═'*56}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NIGHTWATCH blackbox post-event analysis")
    parser.add_argument("--db",    default="nightwatch.sqlite",
                        help="Path to SQLite database (default: nightwatch.sqlite)")
    parser.add_argument("--limit", default=20000, type=int,
                        help="Max rows to load (default: 20000)")
    parser.add_argument("--out",   default="nightwatch_analysis.png",
                        help="Output image path (default: nightwatch_analysis.png)")
    args = parser.parse_args()

    print(f"[BLACKBOX] Cargando {args.db}  (limit={args.limit:,}) ...")
    try:
        rows = load_db(args.db, args.limit)
    except Exception as e:
        print(f"[ERROR] No se pudo abrir la base de datos: {e}")
        sys.exit(1)

    if not rows:
        print("[WARN] La base de datos está vacía. Ejecuta el sistema primero.")
        sys.exit(0)

    partitioned = by_class(rows)
    print_summary(rows, partitioned)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle("NIGHTWATCH  ·  Análisis Post-Evento  ·  KGL v6.0",
                 color="#00ff88", fontsize=13, fontweight="bold",
                 fontfamily="monospace", y=0.98)

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.42, wspace=0.30,
                           left=0.06, right=0.97,
                           top=0.93,  bottom=0.07)

    ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
    ax_phase = fig.add_subplot(gs[0, 1])
    ax_omega = fig.add_subplot(gs[1, 0])
    ax_heat  = fig.add_subplot(gs[1, 1])

    draw_polar_scatter    (ax_polar, partitioned)
    draw_phase_space      (ax_phase, partitioned)
    draw_omega_distribution(ax_omega, partitioned)
    draw_x_heatmap        (ax_heat,  partitioned["X"])

    # Footnote
    fig.text(0.5, 0.01,
             f"Generado desde {args.db}  ·  {len(rows):,} contactos  ·  "
             f"thresholds: γ=9.21  ω_hyper=8°/s  ω_static=0.005°/s",
             ha="center", color="#2a4a3a", fontsize=7, fontfamily="monospace")

    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[DONE] Análisis guardado en: {args.out}")

    # Try to display interactively if a display is available
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
