#!/usr/bin/env python3
"""
two_device_fl.py
----------------
Interactive two-device federated learning demo using FMI daily temperature data.

The script walks through four steps and pauses for a key press between each,
updating the scatter plots so you can see what is happening at every stage.

  Step 1 — Local training
      Each device fits a regression model (tmin → tmax) on its private data.

  Step 2 — Synthetic feature generation
      Each device fits a 1-D Gaussian N(μ, σ²) to its local tmin values and
      draws N_SYNTH synthetic tmin samples from that distribution.

  Step 3 — Pseudo-label exchange
      Each device applies its local model to its synthetic features, producing
      pseudo-labels, and sends the (tmin_synth, tmax_pseudo) pairs to the
      other device.  The plot shows each device's local data together with the
      received pseudo-labeled points from its peer.

  Step 4 — Augmented re-training
      Each device pools its local data with the received pseudo-labeled set and
      re-fits a fresh model of the same type.  The updated decision boundary is
      shown alongside both data sources.

Devices / data
--------------
  Device 1 : Rovaniemi,  Jan–Feb 2025  (cold, high variance)
  Device 2 : Mariehamn,  Jul–Aug 2020  (warm, low variance)

The two devices cover very different temperature regimes, so the synthetic
data exchange is informative: each device receives pseudo-labeled samples
that extend its coverage into a regime it has never observed locally.
"""

import requests
import xml.etree.ElementTree as ET
import datetime

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ── Hyper-parameters ─────────────────────────────────────────────────────────

N_SYNTH  = 40      # synthetic samples drawn per device per iteration
N_ITER   = 3       # number of FL iterations (steps 2-3-4 repeated)
AXIS_MIN = -35     # shared axis limits (°C)
AXIS_MAX =  35

# ── FMI fetch (same parser as fl_temp_demo.py / ReadInDailyMaxMin.py) ────────

NS = {
    "wfs":    "http://www.opengis.net/wfs/2.0",
    "wml2":   "http://www.opengis.net/waterml/2.0",
    "gml":    "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om":     "http://www.opengis.net/om/2.0",
}

def _parse_fmi_xml(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows = []
    for member in root.findall(".//wfs:member", NS):
        location = member.find(".//target:Location", NS)
        station  = location.find(
            './gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]', NS
        ).text
        param = (
            member.find(".//om:observedProperty", NS)
            .get("{http://www.w3.org/1999/xlink}href")
            .split("param=")[1].split("&")[0]
        )
        for pt in member.findall(".//wml2:point", NS):
            t = pt.find(".//wml2:time", NS).text
            if datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").hour != 0:
                continue   # skip intra-day duplicates; keep only daily midnight records
            val = pt.find(".//wml2:value", NS).text
            rows.append({"station": station, "day": t[:10], "parameter": param,
                         "value": float(val) if val != "NaN" else None})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.pivot_table(index=["station", "day"], columns="parameter",
                         values="value", aggfunc="first")
            .reset_index())
    df.columns.name = None
    return df


def fetch_daily(place: str, start: str, end: str) -> pd.DataFrame:
    url = (
        "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
        "&request=getFeature"
        "&storedquery_id=fmi::observations::weather::daily::timevaluepair"
        f"&place={place}"
        f"&starttime={start}T00:00:00Z"
        f"&endtime={end}T00:00:00Z"
        "&parameters=tmin,tmax&timestep=720"
    )
    print(f"  Fetching {place} ({start} → {end}) …")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return _parse_fmi_xml(r.content)


# ── FL network: two-node graph, each node carries a device dict ───────────────

G = nx.Graph()
G.add_node(0, device={
    "label": "Device 1 – Rovaniemi (Jan–Feb 2025)",
    "place": "Rovaniemi",  "start": "2025-01-01",  "end": "2025-02-28",
    "model_cls": lambda: DecisionTreeRegressor(max_depth=3, random_state=42),
    "model_name": "Decision Tree (depth 3)",
    "color_local": "#1f77b4",    # blue  — local data
    "color_synth": "#aec7e8",    # light blue — received pseudo-labeled data
})
G.add_node(1, device={
    "label": "Device 2 – Mariehamn / Åland (Jul–Aug 2020)",
    "place": "Mariehamn",  "start": "2020-07-01",  "end": "2020-08-31",
    "model_cls": lambda: LinearRegression(),
    "model_name": "Linear Regression",
    "color_local": "#d62728",    # red   — local data
    "color_synth": "#f7b6b6",    # light red — received pseudo-labeled data
})
G.add_edge(0, 1)   # single communication link

devices = [G.nodes[n]["device"] for n in G.nodes]

# ── Helpers ───────────────────────────────────────────────────────────────────

t_range = np.linspace(AXIS_MIN, AXIS_MAX, 300).reshape(-1, 1)   # for plotting model curves


def wait(msg: str = ""):
    """Render current figure and pause until the user presses Enter."""
    plt.draw()
    plt.pause(0.05)
    input(f"\n  {'─'*50}\n  {msg}  Press Enter to continue …\n  {'─'*50}\n")


def draw_axes(ax, dev, *, X_local, y_local, model=None,
              X_recv=None, y_recv=None, step_title=""):
    """
    Redraw one subplot from scratch.

    Parameters
    ----------
    X_local, y_local : local (tmin, tmax) observations
    model            : fitted sklearn model (None = don't draw curve)
    X_recv, y_recv   : pseudo-labeled data received from the peer device
                       (None = not yet available)
    step_title       : text shown as subplot title
    """
    ax.cla()

    # local observations
    ax.scatter(X_local, y_local,
               color=dev["color_local"], alpha=0.75, s=35, zorder=3,
               label=f"Local data (n={len(X_local)})")

    # received pseudo-labeled data (Step 3 onward)
    if X_recv is not None and len(X_recv) > 0:
        ax.scatter(X_recv, y_recv,
                   color=dev["color_synth"], alpha=0.9, s=35,
                   marker="^", edgecolors="grey", linewidths=0.5, zorder=4,
                   label=f"Received pseudo-labels (n={len(X_recv)})")

    # model curve
    if model is not None:
        ax.plot(t_range, model.predict(t_range),
                color="k", lw=2, zorder=5,
                label=dev["model_name"])

    # diagonal reference line  y = x
    ax.plot([AXIS_MIN, AXIS_MAX], [AXIS_MIN, AXIS_MAX],
            color="grey", lw=0.8, linestyle="--", alpha=0.5, zorder=1)

    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("tmin (°C)")
    ax.set_ylabel("tmax (°C)")
    ax.set_title(f"{dev['label']}\n{step_title}", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)


# ── Load data ─────────────────────────────────────────────────────────────────

print("\nLoading FMI data …")
for dev in devices:
    df = fetch_daily(dev["place"], dev["start"], dev["end"])
    df = df.dropna(subset=["tmin", "tmax"]).reset_index(drop=True)
    dev["X"] = df["tmin"].values.reshape(-1, 1)   # feature  (n, 1)
    dev["y"] = df["tmax"].values                  # target   (n,)
    print(f"  {dev['label']}: {len(df)} valid days")

# ── Set up persistent figure ──────────────────────────────────────────────────

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Two-Device Federated Learning Demo", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Local model training
# Each device independently fits a regression model on its private data.
# No data leaves the device at this point.
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("STEP 1 — Local model training")
print("═"*55)

for dev in devices:
    dev["model"] = dev["model_cls"]()          # fresh model instance
    dev["model"].fit(dev["X"], dev["y"])
    rmse = mean_squared_error(dev["y"], dev["model"].predict(dev["X"])) ** 0.5
    print(f"  {dev['label']}")
    print(f"    Model : {dev['model_name']}  |  Train RMSE: {rmse:.2f} °C")
    dev["rmse_local"] = rmse

for dev, ax in zip(devices, axes):
    draw_axes(ax, dev,
              X_local=dev["X"].ravel(), y_local=dev["y"],
              model=dev["model"],
              step_title=f"Step 1: local model  (RMSE={dev['rmse_local']:.2f} °C)")

fig.suptitle("Step 1 — Local Model Training", fontsize=13, fontweight="bold")
wait("Step 1 done.")

rng = np.random.default_rng(seed=0)

for fl_iter in range(1, N_ITER + 1):

    print(f"\n{'█'*55}")
    print(f"  FL ITERATION {fl_iter} / {N_ITER}")
    print(f"{'█'*55}")

    # ── Step 2: Synthetic feature generation ─────────────────────────────────
    # Fit a 1-D Gaussian to the current model's input distribution and draw
    # N_SYNTH synthetic tmin values.  In later iterations the model has been
    # updated by augmentation, so the sampling is always relative to the
    # original local feature distribution (unchanged private data).

    print("\n" + "═"*55)
    print(f"STEP 2 (iter {fl_iter}) — Synthetic feature generation")
    print("═"*55)

    for dev in devices:
        tmin_vals = dev["X"].ravel()
        mu, sigma = tmin_vals.mean(), tmin_vals.std()
        dev["X_synth"] = rng.normal(loc=mu, scale=sigma,
                                    size=N_SYNTH).reshape(-1, 1)
        print(f"  {dev['label']}")
        print(f"    tmin: N(μ={mu:.1f}, σ={sigma:.1f})  →  {N_SYNTH} synthetic samples")

    for dev, ax in zip(devices, axes):
        draw_axes(ax, dev,
                  X_local=dev["X"].ravel(), y_local=dev["y"],
                  model=dev["model"],
                  step_title=f"Iter {fl_iter} · Step 2: synthetic features (orange rug)")
        ax.scatter(dev["X_synth"].ravel(),
                   np.full(N_SYNTH, AXIS_MIN + 1.0),
                   color="darkorange", marker="|", s=80, linewidths=1.5,
                   zorder=6, label=f"Synthetic tmin (n={N_SYNTH})")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Iter {fl_iter}/{N_ITER} — Step 2: Synthetic Feature Generation",
                 fontsize=13, fontweight="bold")
    wait(f"Iter {fl_iter} · Step 2 done.")

    # ── Step 3: Pseudo-label exchange ─────────────────────────────────────────
    # Each device labels its synthetic features with its *current* model
    # (which is the augmented model from the previous iteration, except in
    # iteration 1 where it is the locally trained model).

    print("\n" + "═"*55)
    print(f"STEP 3 (iter {fl_iter}) — Pseudo-label generation and exchange")
    print("═"*55)

    for dev in devices:
        dev["y_pseudo"] = dev["model"].predict(dev["X_synth"])

    # exchange pseudo-labels along each edge in the FL graph
    for n in G.nodes:
        dev = G.nodes[n]["device"]
        neighbours = [G.nodes[nb]["device"] for nb in G.neighbors(n)]
        dev["X_recv"] = np.vstack([nb["X_synth"] for nb in neighbours])
        dev["y_recv"] = np.concatenate([nb["y_pseudo"] for nb in neighbours])

    for dev in devices:
        print(f"  {dev['label']}: received {len(dev['X_recv'])} pseudo-labeled samples")

    for dev, ax in zip(devices, axes):
        draw_axes(ax, dev,
                  X_local=dev["X"].ravel(), y_local=dev["y"],
                  model=dev["model"],
                  X_recv=dev["X_recv"].ravel(), y_recv=dev["y_recv"],
                  step_title=f"Iter {fl_iter} · Step 3: received pseudo-labels (▲)")

    fig.suptitle(f"Iter {fl_iter}/{N_ITER} — Step 3: Pseudo-Label Exchange",
                 fontsize=13, fontweight="bold")
    wait(f"Iter {fl_iter} · Step 3 done.")

    # ── Step 4: Augmented re-training ─────────────────────────────────────────
    # Pool local data with received pseudo-labeled data and re-fit.
    # The model trained here becomes the starting model for the next iteration.

    print("\n" + "═"*55)
    print(f"STEP 4 (iter {fl_iter}) — Augmented re-training")
    print("═"*55)

    for dev in devices:
        rmse_before = mean_squared_error(
            dev["y"], dev["model"].predict(dev["X"])) ** 0.5

        X_aug = np.vstack([dev["X"],      dev["X_recv"]])
        y_aug = np.concatenate([dev["y"], dev["y_recv"]])

        model_new = dev["model_cls"]()
        model_new.fit(X_aug, y_aug)
        rmse_after = mean_squared_error(
            dev["y"], model_new.predict(dev["X"])) ** 0.5

        print(f"  {dev['label']}")
        print(f"    Aug. set: {len(y_aug)} samples  "
              f"(local {len(dev['y'])} + received {len(dev['y_recv'])})")
        print(f"    Local RMSE: {rmse_before:.2f} → {rmse_after:.2f} °C")

        dev["model_prev"] = dev["model"]   # keep for overlay
        dev["model"]      = model_new      # becomes current model
        dev["rmse_before"] = rmse_before
        dev["rmse_after"]  = rmse_after

    for dev, ax in zip(devices, axes):
        draw_axes(ax, dev,
                  X_local=dev["X"].ravel(), y_local=dev["y"],
                  model=dev["model"],
                  X_recv=dev["X_recv"].ravel(), y_recv=dev["y_recv"],
                  step_title=(f"Iter {fl_iter} · Step 4: augmented model  "
                               f"RMSE {dev['rmse_before']:.2f}→{dev['rmse_after']:.2f} °C"))
        ax.plot(t_range, dev["model_prev"].predict(t_range),
                color="grey", lw=1.2, linestyle=":", alpha=0.6,
                label=f"Pre-iter model (RMSE={dev['rmse_before']:.2f} °C)")
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Iter {fl_iter}/{N_ITER} — Step 4: Augmented Re-Training",
                 fontsize=13, fontweight="bold")

    if fl_iter < N_ITER:
        wait(f"Iter {fl_iter} · Step 4 done.")
    else:
        plt.draw()
        plt.pause(0.05)

# ── Save final figure ─────────────────────────────────────────────────────────

out = "two_device_fl.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFinal figure saved → {out}")
plt.ioff()
plt.show()
