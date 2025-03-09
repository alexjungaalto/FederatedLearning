import socket
import threading
import pickle
import time
import random
import sys


class Worker:
    def __init__(self, worker_id, local_a, neighbors, lr, alpha, speed):
        self.worker_id = worker_id
        self.local_a = local_a
        self.neighbors = set(neighbors)  # Ensure unique neighbors
        self.lr = lr
        self.alpha = alpha
        self.speed = speed
        self.w = 0.0  # Worker parameter
        self.neighbor_values = {}  # Store latest received values from neighbors
        self.lock = threading.Lock()  # Ensure thread safety

        self.listen_port = 5000 + worker_id  # Define the port for listening
        self.update_interval = 3.0 / speed  # Update interval based on speed

    def send_update(self, neighbor, port):
        """Send the latest parameter along with the sender's ID."""
        param_packet = (self.worker_id, self.w)  # Include worker ID
        max_retries = 5

        for attempt in range(max_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(("127.0.0.1", port))
                    s.sendall(pickle.dumps(param_packet))  # Send as a tuple
                return
            except Exception as e:
                print(f"Worker {self.worker_id}: Failed to send update to {neighbor} on port {port} "
                      f"(Attempt {attempt+1}/{max_retries}) - {e}")
                time.sleep(0.5)

    def listen_for_updates(self):
        """Listen for updates and verify sender identity."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", self.listen_port))
            s.listen()
            print(f"Worker {self.worker_id} listening on port {self.listen_port}...")

            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if data:
                        try:
                            sender_id, received_w = pickle.loads(data)  # Extract ID and value
                            if sender_id in self.neighbors:  # Ensure it's a real neighbor
                                with self.lock:
                                    self.neighbor_values[sender_id] = received_w  # Store latest value
                                print(f"Worker {self.worker_id}: Received update from neighbor {sender_id}: {received_w}")
                            else:
                                print(f"Worker {self.worker_id}: Ignored message from non-neighbor {sender_id}")
                        except Exception as e:
                            print(f"Worker {self.worker_id}: Failed to decode message - {e}")

    def run(self):
        """Main loop: listen, compute updates, and send updates."""
        # Start listening thread
        threading.Thread(target=self.listen_for_updates, daemon=True).start()

        # Introduce slight startup delay to allow neighbors to initialize
        time.sleep(random.uniform(1, 3))

        while True:
            # Compute local gradient
            gradient = 2 * (self.w - self.local_a)

            # Compute TV gradient: (w - w_avg) term
            with self.lock:
                current_neighbor_values = self.neighbor_values.copy()  # Copy values for consistent logging
                if current_neighbor_values:
                    w_avg = sum(current_neighbor_values.values()) / len(current_neighbor_values)
                    tv_gradient = 2 * self.alpha * len(current_neighbor_values) * (self.w - w_avg)
                else:
                    tv_gradient = 0  # No neighbors yet, avoid division by zero

            # Print neighbor values used for the update
            print(f"Worker {self.worker_id}: Neighbor values used for update: {current_neighbor_values}")

            # Gradient descent update
            self.w -= self.lr * (gradient + tv_gradient)

            print(f"Worker {self.worker_id}: Updated w = {self.w:.4f} (Local grad: {gradient:.4f}, "
                  f"TV grad: {tv_gradient:.4f}, lr={self.lr}, alpha={self.alpha}, speed={self.speed})", flush=True)

            # Send updates to neighbors
            for neighbor in self.neighbors:
                neighbor_port = 5000 + neighbor
                self.send_update(neighbor, neighbor_port)

            time.sleep(self.update_interval)  # Control update frequency


if __name__ == "__main__":
    print(sys.argv)
    worker_id = int(sys.argv[1])  # Unique ID
    local_a = float(sys.argv[2])  # Local offset a_i
    lr = float(sys.argv[3])  # Learning rate
    alpha = float(sys.argv[4])  # TV regularization weight (same for all)
    speed = float(sys.argv[5])  # Speed of worker updates
    neighbors = list(map(int, sys.argv[6:]))  # Neighboring worker IDs

    print(f"Worker {worker_id} with neighbors: {neighbors}")

    worker_instance = Worker(worker_id, local_a, neighbors, lr, alpha, speed)
    worker_instance.run()
