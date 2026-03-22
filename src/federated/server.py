"""
server.py
Flower server setup and simulation launcher.
Orchestrates federated training across all hospital nodes.
"""

import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters

from src.models.densenet import build_model, get_parameters
from src.federated.strategy import FedProxStrategy


def get_initial_parameters(config: dict, device: torch.device):
    """
    Initialise server with pretrained DenseNet121 weights.
    All clients start from the same global model.
    """
    model = build_model(config, device)
    parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays_to_parameters(parameters)


def build_strategy(config: dict, device: torch.device) -> FedProxStrategy:
    """
    Builds the FedProx strategy with all federation config.
    """
    fed_cfg = config["federated"]

    initial_params = get_initial_parameters(config, device)

    strategy = FedProxStrategy(
        config=config,
        device=device,
        save_dir=config["results"]["checkpoints_dir"],
        early_stopping_patience=5,
        # Flower FedAvg base params
        fraction_fit=fed_cfg["fraction_fit"],
        fraction_evaluate=fed_cfg["fraction_evaluate"],
        min_fit_clients=fed_cfg["min_fit_clients"],
        min_evaluate_clients=fed_cfg["min_evaluate_clients"],
        min_available_clients=fed_cfg["num_clients"],
        initial_parameters=initial_params,
    )

    return strategy


def run_federated_simulation(
    config: dict,
    client_fn,
    device: torch.device,
) -> dict:
    """
    Launches Flower in-process simulation.
    All nodes run in the same process (no networking needed).

    Args:
        config    : full yaml config
        client_fn : factory function → HospitalClient
        device    : torch.device

    Returns:
        history dict with per-round metrics
    """
    fed_cfg = config["federated"]
    strategy = build_strategy(config, device)

    print("\n" + "=" * 55)
    print("       FEDERATED LEARNING SIMULATION START")
    print("=" * 55)
    print(f"  Nodes        : {fed_cfg['num_clients']}")
    print(f"  Rounds       : {fed_cfg['num_rounds']}")
    print(f"  Strategy     : {fed_cfg['strategy'].upper()}")
    print(f"  Local epochs : {config['training']['epochs_per_round']}")
    print(f"  DP enabled   : {config['privacy']['enabled']}")
    print("=" * 55 + "\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=fed_cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=fed_cfg["num_rounds"]),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 1.0 / fed_cfg["num_clients"],  # share GPU
        },
    )

    history = strategy.get_history()

    print("\n" + "=" * 55)
    print("       FEDERATED TRAINING COMPLETE")
    print(f"  Best Val Accuracy : {strategy.best_accuracy:.2f}%")
    print("=" * 55 + "\n")

    return history
