#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 21:30:27 2025

@author: junga1
"""

import socket
import threading
import time
import json
import numpy as np
import argparse

class AsyncGDWorker:
    def __init__(self, worker_id, port, neighbors, lr=0.1, update_interval=1.0):
        self.worker_id = worker_id
        self.port = port
        self.neighbors = neighbors  # List of (neighbor_ip, neighbor_port)
        self.lr = lr
        self.update_interval = update_interval

        self.model_params = np.random.randn(2)  # Initial model weights
        self.received_updates = {}

        # Start server thread
        threading.Thread(target=self.listen_for_updates, daemon=True).start()

    def listen_for_updates(self):
        """Listen for parameter updates from neighbors."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("0.0.0.0", self.port))
        server_socket.listen(5)
        print(f"[Worker {self.worker_id}] Listening on port {self.port}")

        while True:
            conn, _ = server_socket.accept()
            data = conn.recv(1024).decode()
            conn.close()

            if data:
                message = json.loads(data)
                sender_id = message["worker_id"]
                received_params = np.array(message["parameters"])
                print(f"[Worker {self.worker_id}] Received update from Worker {sender_id}: {received_params}")

                self.received_updates[sender_id] = received_params

    def send_update(self):
        """Send the current parameters to all neighbors."""
        message = json.dumps({"worker_id": self.worker_id, "parameters": self.model_params.tolist()})

        for neighbor in self.neighbors:
            try:
                neighbor_ip, neighbor_port = neighbor
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.connect((neighbor_ip, neighbor_port))
                client_socket.sendall(message.encode())
                client_socket.close()
                print(f"[Worker {self.worker_id}] Sent update to {neighbor_ip}:{neighbor_port}")
            except Exception as e:
                print(f"[Worker {self.worker_id}] Failed to send update to {neighbor}: {e}")

    def compute_gradient(self):
        """Simulated gradient computation (e.g., L = (w1 - 3)^2 + (w2 + 2)^2)."""
        return 2 * (self.model_params - np.array([3, -2]))

    def update_model(self):
        """Perform gradient update based on received parameters."""
        if not self.received_updates:
            return  # No new updates

        # Average received model parameters
        new_params = np.mean(list(self.received_updates.values()), axis=0)

        # Compute local gradient
        gradient = self.compute_gradient()

        # Apply update step
        self.model_params = new_params - self.lr * gradient
        print(f"[Worker {self.worker_id}] Updated model parameters: {self.model_params}")

        # Clear received updates
        self.received_updates.clear()

    def run(self):
        """Main loop: periodically update model and send parameters."""
        while True:
            time.sleep(self.update_interval)

            # Update model if new parameters are available
            self.update_model()

            # Send new parameters to neighbors
            self.send_update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous GD Worker")
    parser.add_argument("--id", type=int, required=True, help="Worker ID")
    parser.add_argument("--port", type=int, required=True, help="Listening port")
    parser.add_argument("--neighbors", nargs="+", help="List of neighbors in format IP:PORT", required=True)
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")

    args = parser.parse_args()

    # Parse neighbors
    neighbor_list = [tuple(neighbor.split(":")) for neighbor in args.neighbors]
    neighbor_list = [(ip, int(port)) for ip, port in neighbor_list]

    worker = AsyncGDWorker(worker_id=args.id, port=args.port, neighbors=neighbor_list, lr=args.lr, update_interval=args.interval)
    worker.run()
