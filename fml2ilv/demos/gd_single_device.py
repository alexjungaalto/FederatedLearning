#!/usr/bin/env python3
"""
gd_single_device.py
-------------------
Single-device gradient descent demo: predict tmax from tmin using a linear
model  tmax ≈ w0 + w1 * tmin  trained on Helsinki daily temperature data.

Two plots are updated after every key press:
  Left  — scatter of data with the current hypothesis line
  Right — loss landscape L(w0, w1) as filled contours with the GD trajectory

Each Enter press performs one gradient descent step.
Press Ctrl-C or close the window to quit.

Connects the slide "Gradient Step as an Affine Map" to a concrete example:
  w^{(t+1)} = F * w^{(t)} + b
where F = I - (2η/m) X^T X  and  b = (2η/m) X^T y.
Model parameters: w = (w1, w2) with w1 = intercept, w2 = slope.
"""

import os
import csv
import datetime
import requests
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Hyper-parameters ─────────────────────────────────────────────────────────

MAX_STEPS   = 200       # hard cap (prevents infinite loop if user keeps pressing)
# ETA (η) is computed from the data after feature normalisation (see below)

PLACE       = "Helsinki"
START       = "2024-01-01"
END         = "2024-12-31"

DEMO_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(DEMO_DIR, "gd_helsinki.csv")

# ── FMI fetch / cache ─────────────────────────────────────────────────────────

NS = {
    "wfs":    "http://www.opengis.net/wfs/2.0",
    "wml2":   "http://www.opengis.net/waterml/2.0",
    "gml":    "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om":     "http://www.opengis.net/om/2.0",
}

def _parse(xml_bytes):
    root = ET.fromstring(xml_bytes)
    rows = []
    for member in root.findall(".//wfs:member", NS):
        loc  = member.find(".//target:Location", NS)
        stn  = loc.find('./gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]', NS).text
        href = member.find(".//om:observedProperty", NS).get("{http://www.w3.org/1999/xlink}href", "")
        param = href.split("param=")[1].split("&")[0] if "param=" in href else ""
        for pt in member.findall(".//wml2:point", NS):
            t = pt.find(".//wml2:time", NS).text
            if datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").hour != 0:
                continue
            v = pt.find(".//wml2:value", NS).text
            rows.append({"station": stn, "day": t[:10], "parameter": param,
                         "value": float(v) if v != "NaN" else None})
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.pivot_table(index=["station","day"], columns="parameter",
                         values="value", aggfunc="first").reset_index())
    df.columns.name = None
    return df

def fetch_or_load():
    """Return (tmin, tmax) arrays, using cache if available."""
    import pandas as pd
    if os.path.exists(CACHE_FILE):
        print(f"Loading from cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
    else:
        url = (
            "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
            "&request=getFeature"
            "&storedquery_id=fmi::observations::weather::daily::timevaluepair"
            f"&place={PLACE}"
            f"&starttime={START}T00:00:00Z"
            f"&endtime={END}T00:00:00Z"
            "&parameters=tmin,tmax&timestep=720"
        )
        print(f"Fetching {PLACE} ({START} → {END}) …")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = _parse(r.content)
        df.to_csv(CACHE_FILE, index=False)
        print(f"  Saved to {CACHE_FILE}")

    df = df.dropna(subset=["tmin","tmax"]).reset_index(drop=True)
    print(f"  {len(df)} valid days")
    return df["tmin"].values, df["tmax"].values

# ── Gradient descent for  L(w) = (1/m)||Xw - y||²  ──────────────────────────
# Model:  tmax ≈ w1 + w2*tmin_norm   (w = [w1, w2])
# Design matrix X has columns [1, tmin_norm], so X ∈ R^{m×2}.

tmin_raw, tmax = fetch_or_load()
m = len(tmin_raw)

# ── Feature normalisation (z-score) ──────────────────────────────────────────
# Standardise tmin to zero mean, unit variance so that both columns of X
# have comparable scale.  This makes X^T X well-conditioned and lets us
# choose the learning rate from the spectrum of X^T X analytically.
tmin_mu    = tmin_raw.mean()
tmin_sigma = tmin_raw.std()
tmin_norm  = (tmin_raw - tmin_mu) / tmin_sigma   # z-scored feature

# Build design matrix with the normalised feature
X = np.column_stack([np.ones(m), tmin_norm])   # (m, 2)
y = tmax                                        # (m,)

# ── Optimal learning rate from the spectrum of X^T X ─────────────────────────
# The Hessian of L is  H = (2/m) X^T X.
# GD converges iff  η < 1 / λ_max(H) = m / (2 λ_max(X^T X)).
# The optimal (fastest-converging) step size is
#   η* = m / (λ_max(X^T X) + λ_min(X^T X))
# which gives contraction rate  κ = (λ_max - λ_min) / (λ_max + λ_min).
eigvals    = np.linalg.eigvalsh(X.T @ X)        # sorted ascending
lam_min    = eigvals[0]
lam_max    = eigvals[-1]
ETA_FRAC = 0.25                                # fraction of optimal (< 1 → suboptimal)
ETA = ETA_FRAC * m / (lam_max + lam_min)   # η = ETA_FRAC * η*
kappa      = lam_max / lam_min                  # condition number
# contraction rate for a scaled step: ||F|| = |1 - ETA_FRAC| when κ=1
rate       = abs(1 - ETA_FRAC)

print(f"  λ_min(X^TX)={lam_min:.2f}  λ_max(X^TX)={lam_max:.2f}")
print(f"  Condition number κ = {kappa:.2f}")
print(f"  Optimal η* = {ETA:.4f}  |  contraction rate = {rate:.4f}")

# ── Reference solution (normal equations) ────────────────────────────────────
w_star, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
L_star = np.mean((X @ w_star - y) ** 2)
print(f"  Optimal weights (norm. space): w1={w_star[0]:.3f}, w2={w_star[1]:.3f}")
print(f"  Optimal loss    : {L_star:.4f}")

def loss(w):
    """Mean squared loss L(w) = (1/m)||Xw - y||²  (normalised feature space)."""
    return np.mean((X @ w - y) ** 2)

def grad(w):
    """Gradient ∇L(w) = (2/m) X^T (Xw - y)."""
    return (2 / m) * X.T @ (X @ w - y)

def gd_step(w):
    """One gradient descent step with learning rate η."""
    return w - ETA * grad(w)

def to_original(w):
    """Convert (w0, w1) in normalised space to (b0, b1) in original tmin space.

    In normalised space:  tmax ≈ w1 + w2 * (tmin - μ) / σ
    In original space:    tmax ≈ (w1 - w2*μ/σ)  +  (w2/σ) * tmin
    """
    b1 = w[1] / tmin_sigma
    b0 = w[0] - w[1] * tmin_mu / tmin_sigma
    return np.array([b0, b1])

# ── Pre-compute loss landscape ────────────────────────────────────────────────
# Grid in normalised weight space, centred on the optimum.

GRID_N     = 120
w0_margin  = 4 * abs(w_star[0])  + 2.0
w1_margin  = 4 * abs(w_star[1])  + 1.0
w0_grid    = np.linspace(w_star[0] - w0_margin, w_star[0] + w0_margin, GRID_N)
w1_grid    = np.linspace(w_star[1] - w1_margin, w_star[1] + w1_margin, GRID_N)
W0, W1     = np.meshgrid(w0_grid, w1_grid)
L_grid     = np.array([[loss(np.array([a, b])) for a in w0_grid] for b in w1_grid])

# ── Initialisation ────────────────────────────────────────────────────────────
# Start away from the optimum (offset in normalised weight space).

w          = w_star + np.array([-w0_margin * 0.7, -w1_margin * 0.7])
trajectory = [w.copy()]

# ── Figure setup ─────────────────────────────────────────────────────────────

plt.ion()
fig, (ax_data, ax_param) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Gradient Descent — Single Device (Helsinki)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Range for hypothesis line in original tmin space
tmin_line = np.linspace(tmin_raw.min() - 2, tmin_raw.max() + 2, 200)

# Original-space optimal weights (for the reference line)
b_star = to_original(w_star)

def draw(step_idx):
    """Redraw both panels for the current weight vector w."""

    # Convert current w (normalised space) to original tmin space for plotting
    b = to_original(w)

    # ── Left: data + hypothesis ───────────────────────────────────────────
    ax_data.cla()
    ax_data.scatter(tmin_raw, tmax, color="#1f77b4", alpha=0.5, s=20, zorder=3,
                    label=f"Helsinki 2024  (m={m})")
    ax_data.plot(tmin_line, b[0] + b[1] * tmin_line,
                 color="crimson", lw=2, zorder=4,
                 label=f"$w_1={w[0]:.2f},\\; w_2={w[1]:.2f}$")
    ax_data.plot(tmin_line, b_star[0] + b_star[1] * tmin_line,
                 color="grey", lw=1.2, linestyle="--", alpha=0.6,
                 label=f"Optimal ($L^*={L_star:.2f}$)")
    ax_data.set_xlabel("tmin (°C)")
    ax_data.set_ylabel("tmax (°C)")
    ax_data.set_title(f"Step {step_idx}  |  "
                      f"$L(w)={loss(w):.3f}$  |  "
                      f"$\\|\\nabla L\\|={np.linalg.norm(grad(w)):.3f}$",
                      fontsize=9)
    ax_data.legend(fontsize=8)
    ax_data.grid(True, alpha=0.25)

    # ── Right: loss landscape + trajectory ───────────────────────────────
    ax_param.cla()
    cf = ax_param.contourf(W0, W1, L_grid, levels=30, cmap="Blues_r", alpha=0.85)
    ax_param.contour(W0, W1, L_grid, levels=30, colors="white",
                     linewidths=0.3, alpha=0.4)

    # trajectory so far
    traj = np.array(trajectory)
    ax_param.plot(traj[:, 0], traj[:, 1],
                  color="crimson", lw=1.5, marker="o", markersize=4,
                  zorder=5, label="GD path")
    ax_param.plot(*w, "ro", markersize=8, zorder=6)          # current position
    ax_param.plot(*w_star, "w*", markersize=12, zorder=7,
                  label=f"$\\hat{{w}}$=({w_star[0]:.1f},{w_star[1]:.1f})")

    ax_param.set_xlabel("$w_1$ (intercept)")
    ax_param.set_ylabel("$w_2$ (slope / tmin coeff)")
    ax_param.set_title("Loss landscape $L(w_1,w_2)$", fontsize=9)
    ax_param.legend(fontsize=8, loc="upper right")

    plt.draw()
    plt.pause(0.05)

# ── Interactive loop ──────────────────────────────────────────────────────────

w_init = w.copy()   # save initial position for restart

while True:
    # reset to initial state
    w          = w_init.copy()
    trajectory = [w.copy()]

    print("\n" + "="*55)
    print(" Gradient Descent Demo")
    print(f"  η = {ETA:.4f}  |  starting at w = [{w[0]:.2f}, {w[1]:.2f}]")
    print("  Enter — next step  |  r + Enter — restart  |  Ctrl-C — quit")
    print("="*55)

    draw(0)

    restart = False
    for step in range(1, MAX_STEPS + 1):
        try:
            key = input(
                f"\n  [step {step}]  L={loss(w):.4f}  ||∇L||={np.linalg.norm(grad(w)):.4f}"
                f"  →  Enter / r: "
            ).strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if key == "r":
            restart = True
            break

        w = gd_step(w)
        trajectory.append(w.copy())
        draw(step)

        g_norm = np.linalg.norm(grad(w))
        print(f"  w1={w[0]:.4f}  w2={w[1]:.4f}  L={loss(w):.4f}  ||∇L||={g_norm:.4f}")
        if g_norm < 1e-4:
            print("  Converged.  (r + Enter to restart, Enter to quit)")
            try:
                restart = input("  → ").strip().lower() == "r"
            except (KeyboardInterrupt, EOFError):
                pass
            break
    else:
        # hit MAX_STEPS
        try:
            restart = input("\n  Max steps reached.  r + Enter to restart: ").strip().lower() == "r"
        except (KeyboardInterrupt, EOFError):
            pass

    if not restart:
        break

# ── Save final figure ─────────────────────────────────────────────────────────

out = os.path.join(DEMO_DIR, "gd_single_device.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.ioff()
plt.show()