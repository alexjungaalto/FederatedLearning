from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def get_strategy(f_fit: float, f_evaluate: float):
    """Return a standard FedAvg strategy.
    
    Use the specified `f_fit` and `f_evaluate` to control the fraction
    of the available/connected clients that should be enrolled in a round
    of fit() and evaluate()."""
    # Define strategy with specified sampling fractions
    return fl.server.strategy.FedAvg(fraction_fit=f_fit,
                                     fraction_evaluate=f_evaluate,
                                     evaluate_metrics_aggregation_fn=weighted_average)


def main():

    # Create strategy
    strategy = get_strategy(f_fit=1.0, f_evaluate=1.0)

    # Start Flower server
    # Clients will need to know the IP address of the machine
    # running the server. They should connect o port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3), # Note we do just 3 rounds of FL
        strategy=strategy,
    )

if __name__ == "__main__":
    main()