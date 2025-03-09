import threading
import time
import json
import queue

# Define network topology
TOPOLOGY = {
    1: [2, 3, 5],  # Worker 1 now explicitly connects to Worker 5 to ensure reliable messaging
    2: [1, 4],
    3: [1, 4],
    4: [2, 3],
    5: [1]  # Worker 5 only connects to Worker 1
}

# Simulation parameters
ETA = 0.01  # Learning rate
SIMULATION_STEPS = 200
INITIAL_WEIGHTS = {worker_id: 0.0 for worker_id in TOPOLOGY}

# Message queues for inter-worker communication (Simulating IP sockets)
message_queues = {worker_id: queue.Queue() for worker_id in TOPOLOGY}
global_barrier = threading.Barrier(len(TOPOLOGY), action=lambda: reset_barrier())  # Barrier for sync

# Barrier reset function
def reset_barrier():
    """ Resets received messages at each barrier sync """
    for worker_id in TOPOLOGY:
        with message_queues[worker_id].mutex:
            message_queues[worker_id].queue.clear()

# Worker function (Synchronous Gradient Descent with strict barriers)
def worker(worker_id):
    global INITIAL_WEIGHTS
    w = INITIAL_WEIGHTS[worker_id]
    target_value = worker_id * 2  # Each worker has a different f_i

    for iteration in range(SIMULATION_STEPS):
        time.sleep(0.05)
        # Step 1: Send current weight to all neighbors
        message = json.dumps({"worker_id": worker_id, "weight": w, "iteration": iteration})
        for neighbor in TOPOLOGY[worker_id]:
            message_queues[neighbor].put(message)

        # Step 2: Collect messages from all expected neighbors
        received_values = {}
        retries = 0
        MAX_RETRIES = 5

        while len(received_values) < len(TOPOLOGY[worker_id]):
            try:
                message = json.loads(message_queues[worker_id].get(timeout=2))  # Fetch messages with retry
                sender_id = message["worker_id"]
                weight = message["weight"]
                msg_iteration = message["iteration"]

                if msg_iteration == iteration:
                    received_values[sender_id] = weight

            except queue.Empty:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"Worker {worker_id}: RETRY {retries}/{MAX_RETRIES} - Waiting for missing messages at iteration {iteration}")
                else:
                    print(f"Worker {worker_id}: WARNING! Messages missing at iteration {iteration}, proceeding with previous weight.")
                    if received_values:  # If at least some updates were received, proceed with those
                        break  
                    else:
                        print(f"Worker {worker_id}: WAITING for at least one message...")  # Prevents workers from running with no data
                        time.sleep(0.5)  # Wait and retry fetching messages

        # Step 3: Compute update if at least one valid update is received
        if received_values:
            avg_neighbor_w = sum(received_values.values()) / len(received_values)
            grad = (w - target_value) + 100 * (w - avg_neighbor_w)
            w -= ETA * grad

        # Log update
        print(f"\n Worker {worker_id}: Iter {iteration}, w = {w}, neighbors = {TOPOLOGY[worker_id]}")

        # Step 4: Synchronization Barrier (Ensuring all workers stay in sync)
        try:
            global_barrier.wait(timeout=5)  # Ensure no worker waits forever
        except threading.BrokenBarrierError:
            print(f"Worker {worker_id}: WARNING! Barrier broken at iteration {iteration}")

# Start simulation
threads = []
for worker_id in TOPOLOGY:
    thread = threading.Thread(target=worker, args=(worker_id,))
    thread.start()
    threads.append(thread)

# Wait for all workers to finish
for thread in threads:
    thread.join()
