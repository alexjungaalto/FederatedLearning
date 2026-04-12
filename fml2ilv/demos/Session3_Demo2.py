#!/usr/bin/env python3
"""
Session3_Demo2.py
-----------------
Same as Session3_Demo1.py but with feature normalization (z-score).
Demonstrates how standardizing tmin reduces the condition number κ
and the contraction rate, making GD converge much faster.

Two plots are updated after every key press:
  Left  — scatter of data with the current hypothesis line
  Right — loss landscape L(w1, w2) as filled contours with the GD trajectory

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

# ── Hyper-parameters ─────────────────────────────────────────────────────────

MAX_STEPS   = 200       # hard cap (prevents infinite loop if user keeps pressing)
# ETA (η) is computed from the spectrum of X^T X (see below)

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
# Model:  tmax ≈ w1 + w2 * tmin   (w = [w1, w2])
# Design matrix X has columns [1, tmin], so X ∈ R^{m×2}.

tmin, tmax = fetch_or_load()
m = len(tmin)

# Feature normalization (z-score): x̃ = (x - μ) / σ
# This makes X^T X ≈ m I, so κ ≈ 1 and GD converges fast.
tmin_mean  = tmin.mean()
tmin_std   = tmin.std()
tmin_norm  = (tmin - tmin_mean) / tmin_std      # zero mean, unit variance

# Build design matrix with normalized feature
X = np.column_stack([np.ones(m), tmin_norm])    # (m, 2) — column of 1s for intercept
y = tmax                                         # (m,) — target vector

# ── Learning rate from the spectrum of X^T X ─────────────────────────────────
# The Hessian of L is  H = (2/m) X^T X.
# GD converges iff  η < 1 / λ_max(H) = m / (2 λ_max(X^T X)).
# The optimal (fastest-converging) step size is
#   η* = m / (λ_max(X^T X) + λ_min(X^T X))
# which gives contraction rate  κ_rate = (κ - 1) / (κ + 1).
eigvals    = np.linalg.eigvalsh(X.T @ X)        # sorted ascending
lam_min    = eigvals[0]
lam_max    = eigvals[-1]
ETA_FRAC   = 0.25                                # fraction of optimal (< 1 → safer)
ETA        = ETA_FRAC * m / (lam_max + lam_min)  # η = ETA_FRAC * η*
kappa      = lam_max / lam_min                    # condition number of X^T X
rate       = (kappa - 1) / (kappa + 1)            # contraction rate at optimal η*

print(f"  λ_min(X'X) = {lam_min:.2f}   λ_max(X'X) = {lam_max:.2f}")
print(f"  Condition number κ = {kappa:.2f}")
print(f"  η = {ETA:.6f}  (ETA_FRAC={ETA_FRAC} of optimal)")
print(f"  Contraction rate ≈ {rate:.4f}")

# ── Reference solution (normal equations) ────────────────────────────────────
w_star, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
L_star = np.mean((X @ w_star - y) ** 2)
print(f"  Optimal weights: w1={w_star[0]:.3f}, w2={w_star[1]:.3f}")
print(f"  Optimal loss   : {L_star:.4f}")

def loss(w):
    """Mean squared loss L(w) = (1/m)||Xw - y||²."""
    return np.mean((X @ w - y) ** 2)

def grad(w):
    """Gradient ∇L(w) = (2/m) X^T (Xw - y)."""
    return (2 / m) * X.T @ (X @ w - y)

def gd_step(w):
    """One gradient descent step with learning rate η."""
    return w - ETA * grad(w)

# ── Pre-compute loss landscape ────────────────────────────────────────────────
# Grid in weight space, centred on the optimum.

GRID_N     = 120
w1_margin  = 4 * abs(w_star[0])  + 2.0
w2_margin  = 4 * abs(w_star[1])  + 1.0
w1_grid    = np.linspace(w_star[0] - w1_margin, w_star[0] + w1_margin, GRID_N)
w2_grid    = np.linspace(w_star[1] - w2_margin, w_star[1] + w2_margin, GRID_N)
W1g, W2g   = np.meshgrid(w1_grid, w2_grid)
L_grid     = np.array([[loss(np.array([a, b])) for a in w1_grid] for b in w2_grid])

# ── Initialisation ────────────────────────────────────────────────────────────
# Start away from the optimum.

w          = w_star + np.array([-w1_margin * 0.7, -w2_margin * 0.7])
trajectory = [w.copy()]

# ── Figure setup ─────────────────────────────────────────────────────────────

plt.ion()
fig, (ax_data, ax_param) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Gradient Descent — Single Device (Helsinki, normalized)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])

# Range for hypothesis line (original tmin space)
tmin_line = np.linspace(tmin.min() - 2, tmin.max() + 2, 200)
tmin_line_norm = (tmin_line - tmin_mean) / tmin_std

def draw(step_idx):
    """Redraw both panels for the current weight vector w."""

    # ── Left: data + hypothesis (original tmin space) ────────────────────
    ax_data.cla()
    ax_data.scatter(tmin, tmax, color="#1f77b4", alpha=0.5, s=20, zorder=3,
                    label=f"Helsinki 2024  (m={m})")
    ax_data.plot(tmin_line, w[0] + w[1] * tmin_line_norm,
                 color="crimson", lw=2, zorder=4,
                 label=f"$w_1={w[0]:.2f},\\; w_2={w[1]:.2f}$")
    ax_data.plot(tmin_line, w_star[0] + w_star[1] * tmin_line_norm,
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
    ax_param.contourf(W1g, W2g, L_grid, levels=30, cmap="Blues_r", alpha=0.85)
    ax_param.contour(W1g, W2g, L_grid, levels=30, colors="white",
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
    ax_param.set_ylabel("$w_2$ (slope)")
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
    print(f"  η = {ETA:.6f}  |  starting at w = [{w[0]:.2f}, {w[1]:.2f}]")
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

out = os.path.join(DEMO_DIR, "Session3_Demo2.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.ioff()
plt.show()
