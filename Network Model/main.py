import argparse
import yaml
from src.build_mv import build_network
from src.simulate_mv import run

def main():
    parser = argparse.ArgumentParser(description="Run the EV Smart Charging Simulation.")
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    # Build the network
    print("Building the network...")
    net = build_network(with_der=cfg["network"].get("with_der", True))

    # Run the simulation
    print("Running the simulation...")
    run(cfg)

if __name__ == "__main__":
    main()
