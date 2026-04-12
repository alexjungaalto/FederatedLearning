#!/usr/bin/env python3
"""
Session4_auto.py — Automated Federated k-Means Demo (single process)
---------------------------------------------------------------------
Runs both FL devices in one process with a side-by-side plot that updates
live. Pauses between rounds for narration.

Usage:
    python Session4_auto.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ────────────────────────────────────────────────────────────

K          = 3       # number of clusters
N_SAMPLES  = 100     # data points per device
N_ITER     = 6       # FL rounds
ALPHA      = 100.0     # GTVMin regularization parameter α
A_EDGE     = 1.0     # edge weight A_{i,i'}

# ── Generate local 2D data for both devices ──────────────────────────────────

devices = [
    {"id": 1, "centers": [(-5, 3), (0, 0), (-4, -3)], "color": "#1f77b4"},
    {"id": 2, "centers": [(5, -3), (0, 0), (4, 3)],   "color": "#d62728"},
]

for dev in devices:
    X, _ = make_blobs(n_samples=N_SAMPLES, centers=dev["centers"],
                       cluster_std=1.2, random_state=40 + dev["id"])
    dev["X"] = X
    dev["centroids"]    = None
    dev["nb_centroids"] = None

# ── Plotting ─────────────────────────────────────────────────────────────────

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
CENTROID_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]


def draw(dev, ax, rnd, msg=""):
    ax.cla()
    ax.scatter(dev["X"][:, 0], dev["X"][:, 1],
               s=18, alpha=0.4, c="gray",
               label=f"Device {dev['id']} data")
    if dev["centroids"] is not None:
        for ci in range(len(dev["centroids"])):
            ax.scatter(dev["centroids"][ci, 0], dev["centroids"][ci, 1],
                       s=220, marker="X",
                       color=CENTROID_COLORS[ci % len(CENTROID_COLORS)],
                       edgecolors="black", linewidths=1.5, zorder=5)
        ax.scatter([], [], s=120, marker="X", color="gray",
                   edgecolors="black", label=f"Device {dev['id']} centroids")
    if dev["nb_centroids"] is not None:
        ax.scatter(dev["nb_centroids"][:, 0], dev["nb_centroids"][:, 1],
                   s=160, marker="^", color="orange", edgecolors="black",
                   linewidths=1, zorder=4, alpha=0.85,
                   label="Neighbour centroids (received)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"Device {dev['id']} — Round {rnd}  {msg}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(-9, 9); ax.set_ylim(-7, 7)
    ax.grid(True, alpha=0.25)


def draw_all(rnd, msg=""):
    for dev, ax in zip(devices, axes):
        draw(dev, ax, rnd, msg)
    fig.suptitle(f"Federated k-Means — Round {rnd}/{N_ITER}  "
                 f"(α={ALPHA}, A={A_EDGE})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.draw(); plt.pause(0.05)


# ── Round 0: local k-means only ─────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Federated k-Means ({N_ITER} rounds, press Enter to advance)")
print(f"  α={ALPHA}, A_edge={A_EDGE}, R={max(1, int(ALPHA * A_EDGE))}")
print(f"{'='*55}\n")

print("Round 0 — Local k-means (no communication)")
for dev in devices:
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km.fit(dev["X"])
    dev["centroids"] = km.cluster_centers_.copy()
    print(f"  Device {dev['id']} centroids:\n"
          f"    {np.round(dev['centroids'], 2).tolist()}")

draw_all(0, "(local only)")
input("\n  Press Enter to start FL rounds … ")

# ── FL rounds ────────────────────────────────────────────────────────────────

for rnd in range(1, N_ITER + 1):
    print(f"\n{'─'*55}")
    print(f"  Round {rnd}/{N_ITER}")
    print(f"{'─'*55}")

    for i, dev in enumerate(devices):
        nb = devices[1 - i]

        # READ — get neighbour centroids
        dev["nb_centroids"] = nb["centroids"].copy()

        # UPDATE — augment local data with R copies of neighbour centroids
        # R = α · A_{i,i'} controls coupling strength (from GTVMin)
        # e.g. R=5, k=3 → 15 pseudo-data points added to m local points
        n_repeat = max(1, int(ALPHA * A_EDGE))
        # [nb_centroids] * R repeats the (k×d) array R times in a list,
        # then vstack produces an (m + R·k) × d augmented dataset
        X_aug = np.vstack([dev["X"]] + [dev["nb_centroids"]] * n_repeat)

        km = KMeans(n_clusters=K, n_init=10, random_state=42)
        km.fit(X_aug)
        dev["centroids"] = km.cluster_centers_.copy()

        print(f"  Device {dev['id']}: augmented {N_SAMPLES} + "
              f"{n_repeat * K} pseudo-pts → {len(X_aug)} total")
        print(f"    Centroids: {np.round(dev['centroids'], 2).tolist()}")

    draw_all(rnd)

    if rnd < N_ITER:
        try:
            input(f"\n  Press Enter for round {rnd + 1} (Ctrl-C to quit) … ")
        except (KeyboardInterrupt, EOFError):
            print("\n  Stopped by user.")
            break

# ── Done ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  FL complete after {N_ITER} rounds")
for dev in devices:
    print(f"  Device {dev['id']} final centroids:")
    print(f"    {np.round(dev['centroids'], 2).tolist()}")
print(f"{'='*55}")

out = os.path.join(SCRIPT_DIR, "session4_auto.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"  Saved → {out}")

plt.ioff()
plt.show()
