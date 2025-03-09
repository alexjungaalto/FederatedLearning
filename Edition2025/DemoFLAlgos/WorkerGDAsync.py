#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 10:03:13 2025

@author: junga1
"""

import socket
import threading
import pickle
import time
import random
import sys

lambda_tv = 0.5  # Weight for total variation term
learning_rate = 0.01
update_interval = 2.0  # Time (seconds) between gradient steps

def send_update(neighbor, port, param):
    """Send updated parameter to a neighbor."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("127.0.0.1", port))
            s.sendall(pickle.dumps(param))
    except Exception as e:
        print(f"Worker {worker_id}: Failed to send update to {neighbor} on port {port} - {e}")

def listen_for_updates(port):
    """Listen for incoming updates from neighbors."""
    global neighbor_values
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
        s.listen()
        print(f"Worker {worker_id} listening on port {port}...")

        while True:
            conn, addr = s.accept()
            with conn:
                data = conn.recv(1024)
                if data:
                    received_w = pickle.loads(data)
                    sender_id = int(addr[1]) - 5000  # Approximate neighbor ID
                    neighbor_values[sender_id] = received_w  # Store latest value

def worker(worker_id, local_a, neighbors):
    global w, neighbor_values
    w = 0.0  # Initialize parameter
    neighbor_values = {n: w for n in neighbors}  # Track latest values from neighbors

    # Start listening thread
    listen_port = 5000 + worker_id
    threading.Thread(target=listen_for_updates, args=(listen_port,), daemon=True).start()

    while True:
        # Compute local gradient
        gradient = 2 * (w - local_a)

        # Compute TV gradient: (w - w_avg) term
        if neighbors:
            w_avg = sum(neighbor_values.values()) / len(neighbors)
            tv_gradient = 2 * lambda_tv * (w - w_avg)
        else:
            tv_gradient = 0

        # Gradient descent update
        w -= learning_rate * (gradient + tv_gradient)

        print(f"Worker {worker_id}: Updated w = {w:.4f} (Local grad: {gradient:.4f}, TV grad: {tv_gradient:.4f})")

        # Send updates to neighbors
        for neighbor in neighbors:
            neighbor_port = 5000 + neighbor
            send_update(neighbor, neighbor_port, w)

        time.sleep(update_interval + random.uniform(0, 1))  # Random delay added to desynchronize

if __name__ == "__main__":
    worker_id = int(sys.argv[1])  # Unique ID
    local_a = float(sys.argv[2])  # Local offset a_i
    neighbors = list(map(int, sys.argv[3:]))  # Neighboring worker IDs

    worker(worker_id, local_a, neighbors)
