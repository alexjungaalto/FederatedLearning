#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:21:02 2025

@author: junga1
"""

import threading
import time
import json
import queue
import random

# Define network topology
TOPOLOGY = {
    1: [2, 3, 5],  # Worker 1 explicitly connects to Worker 5 to ensure message reliability
    2: [1, 4],
    3: [1, 4],
    4: [2, 3],
    5: [1]  # Worker 5 only connects to Worker 1
}

# Simulation parameters
ETA = 0.1  # Learning rate
SIMULATION_STEPS = 20
INITIAL_WEIGHTS = {worker_id: 0.0 for worker_id in TOPOLOGY}

# Message queues for inter-worker communication (Simulating IP sockets)
message_queues = {worker_id: queue.Queue() for worker_id in TOPOLOGY}

# Worker function (Asynchronous Gradient Descent with Message Retries)
def worker(worker_id):
    global INITIAL_WEIGHTS
    w = INITIAL_WEIGHTS[worker_id]
    target_value = worker_id * 2  # Each worker has a different f_i

    for iteration in range(SIMULATION_STEPS):
        # Introduce random delays to simulate asynchrony
        time.sleep(random.uniform(0.1, 0.5))

        # Step 1: Send current weight to all neighbors (with retries)
        message = json.dumps({"worker_id": worker_id, "weight": w, "iteration": iteration})
        for neighbor in TOPOLOGY[worker_id]:
            for _ in range(3):  # Retry sending up to 3 times
                message_queues[neighbor].put(message)
                time.sleep(0.05)  # Small delay before retrying

        # Step 2: Collect messages from available neighbors (No waiting!)
        received_values = {}
        retries = 0
        MAX_RETRIES = 5

        while not message_queues[worker_id].empty():  # Process all available messages
            try:
                message = json.loads(message_queues[worker_id].get_nowait())  # Fetch messages asynchronously
                sender_id = message["worker_id"]
                weight = message["weight"]
                msg_iteration = message["iteration"]

                received_values[sender_id] = weight

            except queue.Empty:
                break  # No more messages

        # Step 3: Ensure Worker 5 always receives at least one message before updating
        if worker_id == 5 and not received_values:  # Worker 5 has only one neighbor
            print(f"Worker 5: RETRYING! No message received at iteration {iteration}")
            time.sleep(0.2)  # Small delay to wait for messages
            continue  # Retry fetching messages before proceeding

        # Step 4: Compute update using available messages (Proceed even with missing data)
        if received_values:
            avg_neighbor_w = sum(received_values.values()) / len(received_values)
            grad = (w - target_value) + 2 * (w - avg_neighbor_w)
            w -= ETA * grad

        # Log update
        print(f"Worker {worker_id}: Iter {iteration}, w = {w}, neighbors = {TOPOLOGY[worker_id]}")

# Start simulation
threads = []
for worker_id in TOPOLOGY:
    thread = threading.Thread(target=worker, args=(worker_id,))
    thread.start()
    threads.append(thread)

# Wait for all workers to finish
for thread in threads:
    thread.join()
