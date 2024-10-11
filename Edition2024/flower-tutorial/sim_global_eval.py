import flwr as fl
import warnings
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from client import Net, set_parameters, test, apply_transforms, get_client_fn
from server import get_strategy

warnings.simplefilter(action='ignore', category=FutureWarning)

NUM_CLIENTS = 1000

def get_evaluate_fn(centralized_testset):
    """Return a function that will be executed by the strategy
    to evaluate the quality of the global model after aggregation."""

    def evaluate(server_round, parameters, config):
        """Use the entire MNIST test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate model and apply parameters
        model = Net()
        set_parameters(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_transforms)
        # Evaluate on test set
        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device, disable_tqdm=False)

        return loss, {"accuracy": accuracy}

    return evaluate


def main():

    # Get client_fn and strategy
    client_fn = get_client_fn(total_partitions=NUM_CLIENTS, disable_tqdm=True)
    strategy = get_strategy(f_fit=0.1, f_evaluate=0.0)

    # Download MNIST dataset and partition it
    mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
    centralized_testset = mnist_fds.load_split("test")
    # Set evaluate method for strategy
    strategy.evaluate_fn = get_evaluate_fn(centralized_testset)

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )

    print(history)

    # Basic plotting of centralized accuracy
    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [int(data[0]) for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    plt.plot(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.savefig("central_evaluation.png")


if __name__ == "__main__":
    main()