#!/usr/bin/env python3
"""
Session4_node.py — Federated k-Means Demo (two-device communication)
---------------------------------------------------------------------
Run in two separate terminals:

  File mode (default):
    Terminal 1:  python Session4_node.py 1
    Terminal 2:  python Session4_node.py 2

  Socket mode:
    Terminal 1:  python Session4_node.py 1 --socket
    Terminal 2:  python Session4_node.py 2 --socket

Each instance simulates one FL device with local 2D data (Gaussian blobs).
The scatterplot shows local data, own cluster centroids (×), and received
neighbour centroids (▲).

Communication backend:
  --file   (default)  shared JSON file on disk
  --socket            TCP sockets on localhost (Device 1 = server, Device 2 = client)

Press Enter to advance each FL round.
"""

import sys
import os
import json
import time
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SHARED_FILE   = os.path.join(SCRIPT_DIR, "session4_shared_centroids.json")
K             = 3          # number of clusters
N_SAMPLES     = 100        # data points per device
N_ITER        = 5          # FL rounds
ALPHA         = 5.0        # GTVMin regularization parameter α
A_EDGE        = 1.0        # edge weight A_{i,i'}
POLL_INTERVAL = 0.5        # seconds between file polls

# Socket settings
SOCK_HOST     = "127.0.0.1"
SOCK_PORT     = 9740       # TCP port for centroid exchange

# ── Parse arguments ──────────────────────────────────────────────────────────

args = sys.argv[1:]
if not args or args[0] not in ("1", "2"):
    print("Usage: python Session4_node.py <1|2> [--file|--socket]")
    sys.exit(1)

NODE_ID   = int(args[0])
NEIGHBOUR = 2 if NODE_ID == 1 else 1
MODE      = "socket" if "--socket" in args else "file"


# ══════════════════════════════════════════════════════════════════════════════
#  Communication backends
# ══════════════════════════════════════════════════════════════════════════════

# ── File backend ─────────────────────────────────────────────────────────────

def _read_shared():
    if not os.path.exists(SHARED_FILE):
        return {}
    try:
        with open(SHARED_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def file_write(node_id, centroids, iteration):
    """Write centroids to the shared JSON file."""
    data = _read_shared()
    data[f"node_{node_id}_iter_{iteration}"] = centroids.tolist()
    tmp = SHARED_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SHARED_FILE)

def file_read(node_id, iteration, timeout=120):
    """Poll the shared file until the neighbour's entry appears."""
    key = f"node_{node_id}_iter_{iteration}"
    print(f"  [file] Waiting for Device {node_id} (iter {iteration}) …",
          end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        data = _read_shared()
        if key in data:
            print(" received!")
            return np.array(data[key])
        time.sleep(POLL_INTERVAL)
    print(" TIMEOUT!")
    return None


# ── Socket backend ───────────────────────────────────────────────────────────

_sock_conn = None   # persistent TCP connection (set up once)

def _send_array(conn, arr):
    """Send a numpy array over a TCP connection (length-prefixed JSON)."""
    payload = json.dumps(arr.tolist()).encode()
    conn.sendall(struct.pack("!I", len(payload)) + payload)

def _recv_array(conn):
    """Receive a numpy array from a TCP connection."""
    raw_len = b""
    while len(raw_len) < 4:
        chunk = conn.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk
    msg_len = struct.unpack("!I", raw_len)[0]
    data = b""
    while len(data) < msg_len:
        chunk = conn.recv(msg_len - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return np.array(json.loads(data.decode()))

def socket_setup():
    """Establish the TCP connection (Device 1 = server, Device 2 = client)."""
    global _sock_conn
    if NODE_ID == 1:
        # Server: wait for Device 2 to connect
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((SOCK_HOST, SOCK_PORT))
        srv.listen(1)
        print(f"  [socket] Listening on {SOCK_HOST}:{SOCK_PORT} …", flush=True)
        _sock_conn, addr = srv.accept()
        print(f"  [socket] Device 2 connected from {addr}")
        srv.close()
    else:
        # Client: connect to Device 1
        print(f"  [socket] Connecting to {SOCK_HOST}:{SOCK_PORT} …",
              end="", flush=True)
        _sock_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                _sock_conn.connect((SOCK_HOST, SOCK_PORT))
                break
            except ConnectionRefusedError:
                time.sleep(0.5)
        print(" connected!")

def socket_write(node_id, centroids, iteration):
    """Send centroids to the neighbour over TCP."""
    _send_array(_sock_conn, centroids)

def socket_read(node_id, iteration, timeout=120):
    """Receive centroids from the neighbour over TCP."""
    print(f"  [socket] Waiting for Device {node_id} (iter {iteration}) …",
          end="", flush=True)
    _sock_conn.settimeout(timeout)
    try:
        arr = _recv_array(_sock_conn)
        print(" received!")
        return arr
    except (socket.timeout, ConnectionError):
        print(" TIMEOUT!")
        return None


# ── Select backend ───────────────────────────────────────────────────────────

if MODE == "socket":
    write_centroids = socket_write
    read_centroids  = socket_read
else:
    write_centroids = file_write
    read_centroids  = file_read


# ══════════════════════════════════════════════════════════════════════════════
#  Data generation & FL logic (identical for both backends)
# ══════════════════════════════════════════════════════════════════════════════

# ── Generate local 2D data ───────────────────────────────────────────────────

if NODE_ID == 1:
    centers = [(-5, 3), (0, 0), (-4, -3)]
else:
    centers = [(5, -3), (0, 0), (4, 3)]

X_local, _ = make_blobs(n_samples=N_SAMPLES, centers=centers,
                         cluster_std=1.2, random_state=40 + NODE_ID)

print(f"\n{'='*55}")
print(f"  Federated k-Means — Device {NODE_ID}  [{MODE} mode]")
print(f"  {N_SAMPLES} local points, K={K}, α={ALPHA}, A={A_EDGE}")
print(f"{'='*55}\n")


# ── Plotting ─────────────────────────────────────────────────────────────────

plt.ion()
fig, ax = plt.subplots(figsize=(7, 6))
COLORS = ["#1f77b4", "#2ca02c", "#d62728"]

def draw(centroids, nb_centroids, rnd, msg=""):
    ax.cla()
    ax.scatter(X_local[:, 0], X_local[:, 1], s=18, alpha=0.4, c="gray",
               label=f"Device {NODE_ID} data (n={N_SAMPLES})")
    for ci in range(len(centroids)):
        ax.scatter(centroids[ci, 0], centroids[ci, 1],
                   s=220, marker="X", color=COLORS[ci % len(COLORS)],
                   edgecolors="black", linewidths=1.5, zorder=5)
    ax.scatter([], [], s=120, marker="X", color="gray", edgecolors="black",
               label=f"Device {NODE_ID} centroids")
    if nb_centroids is not None:
        ax.scatter(nb_centroids[:, 0], nb_centroids[:, 1], s=160,
                   marker="^", color="orange", edgecolors="black",
                   linewidths=1, zorder=4, alpha=0.85,
                   label=f"Device {NEIGHBOUR} centroids (received)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"Device {NODE_ID} — Round {rnd}  {msg}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(-9, 9); ax.set_ylim(-7, 7)
    ax.grid(True, alpha=0.25)
    plt.tight_layout(); plt.draw(); plt.pause(0.05)


# ── Setup ────────────────────────────────────────────────────────────────────

if MODE == "socket":
    socket_setup()
elif NODE_ID == 1 and os.path.exists(SHARED_FILE):
    os.remove(SHARED_FILE)
    print("  Cleaned shared file.\n")


# ── Round 0: local k-means only ─────────────────────────────────────────────

print("Round 0 — Local k-means (no communication)")
km = KMeans(n_clusters=K, n_init=10, random_state=42)
km.fit(X_local)
centroids = km.cluster_centers_.copy()

print(f"  Centroids:\n{np.round(centroids, 2)}")
draw(centroids, None, 0, "(local only)")

# Share round-0 centroids
write_centroids(NODE_ID, centroids, 0)

input("\n  Press Enter to start FL rounds …\n")


# ── FL rounds ────────────────────────────────────────────────────────────────

for rnd in range(1, N_ITER + 1):
    print(f"\n{'─'*55}")
    print(f"  Round {rnd}/{N_ITER}")
    print(f"{'─'*55}")

    # 1. READ — get neighbour's centroids from previous round
    nb_centroids = read_centroids(NEIGHBOUR, rnd - 1)
    if nb_centroids is None:
        print("  Neighbour not available — skipping round.")
        write_centroids(NODE_ID, centroids, rnd)
        continue

    # 2. UPDATE — augment local data with R copies of neighbour centroids
    #    R = α · A_{i,i'} controls coupling strength (from GTVMin)
    #    e.g. R=5, k=3 → 15 pseudo-data points added to m local points
    n_repeat = max(1, int(ALPHA * A_EDGE))
    # [nb_centroids] * R repeats the (k×d) array R times in a list,
    # then vstack produces an (m + R·k) × d augmented dataset
    X_aug    = np.vstack([X_local] + [nb_centroids] * n_repeat)

    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km.fit(X_aug)
    centroids = km.cluster_centers_.copy()

    print(f"  Augmented: {N_SAMPLES} local + {n_repeat * K} pseudo-points "
          f"= {len(X_aug)}")
    print(f"  Updated centroids:\n{np.round(centroids, 2)}")

    # 3. SHARE — send updated centroids
    write_centroids(NODE_ID, centroids, rnd)

    # 4. DRAW
    draw(centroids, nb_centroids, rnd)

    if rnd < N_ITER:
        input(f"\n  Press Enter for round {rnd + 1} …\n")


# ── Done ─────────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Device {NODE_ID} — complete after {N_ITER} rounds [{MODE} mode]")
print(f"  Final centroids:\n{np.round(centroids, 2)}")
print(f"{'='*55}")

out = os.path.join(SCRIPT_DIR, f"session4_device{NODE_ID}.png")
fig.savefig(out, dpi=100, bbox_inches="tight")
print(f"  Saved → {out}")

if _sock_conn:
    _sock_conn.close()

plt.ioff()
plt.show()