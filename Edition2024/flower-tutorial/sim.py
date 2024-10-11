import flwr as fl
from client import get_client_fn
from server import get_strategy

# Number of total clients in our simulation
# You'll need this many data partitions
NUM_CLIENTS = 1000


def main():

    # Get a function that returns a client associated with one data partition
    client_fn = get_client_fn(total_partitions=NUM_CLIENTS, disable_tqdm=True)

    # Get a strategy that samples 10% of clients for fit and 0% for evaluate
    # You can change thest parameters to see how things work.
    strategy = get_strategy(f_fit=0.1, f_evaluate=0.00)

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}, # each client will use just 1x CPU
        config=fl.server.ServerConfig(num_rounds=10), # we'll run for 10 rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    main()