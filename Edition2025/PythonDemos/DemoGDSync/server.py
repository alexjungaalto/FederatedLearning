import socket
import threading
import pickle
import sys
import signal
import time

class SGDServer:
    def __init__(self, worker_ids, port=5000):
        self.worker_ids = set(worker_ids)  # Expected worker IDs
        self.worker_updates = {}  # Store received updates
        self.lock = threading.Lock()  # Thread safety
        self.port = port
        self.round = 0  # Synchronization round counter
        self.server_socket = None
        self.running = True  # Control server shutdown

    def handle_worker(self, conn, addr):
        """Handles worker updates and synchronizes training steps."""
        try:
            while self.running:
                data = conn.recv(1024)
                if not data:
                    break
                
                worker_id, w_value = pickle.loads(data)

                with self.lock:
                    self.worker_updates[worker_id] = w_value
                    print(f"Server: Received w={w_value:.4f} from Worker {worker_id}")

                    # Wait for all workers before sending sync signal
                    if len(self.worker_updates) == len(self.worker_ids):
                        self.round += 1
                        print(f"\n--- Server: Synchronization Round {self.round} ---")
                        print(f"Server: All workers sent their updates, broadcasting sync signal...\n")

                        # Reset for next iteration
                        self.worker_updates.clear()

                        # Send sync signal to all workers
                        for worker_id in self.worker_ids:
                            self.send_sync_signal(worker_id)
        
        except Exception as e:
            print(f"Server: Error handling worker {worker_id}: {e}")

    def send_sync_signal(self, worker_id, max_retries=5, delay=1):
        """Send synchronization signal to workers with retries if necessary."""
        worker_port = 5000 + worker_id  # Worker ports start from 5001

        for attempt in range(max_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(("127.0.0.1", worker_port))
                    s.sendall(pickle.dumps("SYNC"))  # Send sync signal
                print(f"Server: Sync signal sent to Worker {worker_id} on port {worker_port}")
                return  # Success, exit function
            except Exception as e:
                print(f"Server: Failed to send sync signal to Worker {worker_id} on port {worker_port}, Retrying ({attempt+1}/{max_retries})...")
                time.sleep(delay)  # Wait before retrying

    def start(self):
        """Start server and listen for worker updates."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind(("127.0.0.1", self.port))
            self.server_socket.listen()
            print(f"Server: Listening on port {self.port}... (Press Ctrl+C to stop)")

            while self.running:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_worker, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            self.shutdown()
        finally:
            self.server_socket.close()

    def shutdown(self):
        """Gracefully shutdown the server."""
        print("\nServer shutting down...")
        self.running = False
        self.server_socket.close()
        print("Server: Port released successfully.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python server.py <worker_id_1> <worker_id_2> ...")
        sys.exit(1)

    worker_ids = list(map(int, sys.argv[1:]))  # Convert args to integers
    print(f"Server initialized with workers: {worker_ids}")

    server = SGDServer(worker_ids)

    # Handle termination properly
    signal.signal(signal.SIGINT, lambda sig, frame: server.shutdown())

    server.start()
