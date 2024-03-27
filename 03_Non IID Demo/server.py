import flwr as fl
import sys
import numpy as np

if len(sys.argv) != 3:
    print("ArgumentError - Usage: python3 <script> <server_addr> <server_port>")
    exit(127)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address=str(sys.argv[1]) + ':' + str(sys.argv[2]),
        config=fl.server.ServerConfig(num_rounds=10),
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
)
