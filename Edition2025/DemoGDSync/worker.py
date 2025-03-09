import socket
import threading
import pickle
import time
import sys
import signal


class Worker:
    def __init__(self, worker_id, local_a, neighbors, lr, alpha, speed, server_port=5000):
        self.worker_id = worker_id
        self.local_a = local_a
        self.neighbors = set(neighbors)  # Neighboring worker IDs
        self.lr = lr
        self.alpha = alpha
        self.speed = speed
        self.w = 0.0  # Worker parameter
        self.neighbor_values = {n: 0.0 for n in self.neighbors}  # Latest values from neighbors
        self.received_updates = set()  # Track updates from neighbors
        self.lock = threading.Lock()
        self.running = True  # Control worker shutdown

        self.listen_port = 5000 + worker_id  # Worker listens on 5001, 5002, ...
        self.server_port = server_port  # Server is at 5000

    def send_update_to_server(self):
        """Send the current value of w to the server."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", self.server_port))
                s.sendall(pickle.dumps((self.worker_id, self.w)))
                print(f"Worker {self.worker_id}: Sent update w = {self.w:.4f} to server")
        except Exception as e:
            print(f"Worker {self.worker_id}: Failed to send update to server - {e}")

    def wait_for_sync_signal(self):
        """Wait for synchronization signal from the server, then proceed."""
        while self.running:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", self.listen_port))
                    s.listen()
                    conn, _ = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data and pickle.loads(data) == "SYNC":
                            print(f"Worker {self.worker_id}: Received sync signal, proceeding with update.")
                            return  # Exit the function once sync is received
            except Exception as e:
                print(f"Worker {self.worker_id}: Error receiving sync signal - {e}")
            finally:
                time.sleep(0.1)  # Short delay before restarting the listener

    def run(self):
        """Main loop: ensure worker listens before updating and continues after each round."""
        while self.running:
            self.send_update_to_server()
            self.wait_for_sync_signal()  # Blocks until sync signal is received

            # Compute local gradient
            gradient = 2 * (self.w - self.local_a)

            # Compute TV gradient
            with self.lock:
                if self.neighbor_values:
                    w_avg = sum(self.neighbor_values.values()) / len(self.neighbor_values)
                    tv_gradient = 2 * self.alpha * len(self.neighbor_values) * (self.w - w_avg)
                else:
                    tv_gradient = 0

            # Gradient descent update
            self.w -= self.lr * (gradient + tv_gradient)

            print(f"Worker {self.worker_id}: Updated w = {self.w:.4f}")

            time.sleep(3.0 / self.speed)  # Simulate worker speed

    def stop(self):
        """Gracefully stop the worker and release the port."""
        self.running = False
        print(f"Worker {self.worker_id}: Stopping and releasing port {self.listen_port}.")
        sys.exit(0)


if __name__ == "__main__":
    worker_id = int(sys.argv[1])
    local_a = float(sys.argv[2])
    lr = float(sys.argv[3])
    alpha = float(sys.argv[4])
    speed = float(sys.argv[5])
    neighbors = list(map(int, sys.argv[6:]))

    worker_instance = Worker(worker_id, local_a, neighbors, lr, alpha, speed)

    # Handle termination properly
    signal.signal(signal.SIGINT, lambda sig, frame: worker_instance.stop())

    worker_instance.run()
