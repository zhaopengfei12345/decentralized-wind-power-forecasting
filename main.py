import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from data_loader import load_data
from evaluation import write_metrics
from federated_training import Config, fit


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description="Train the wind power forecasting model.")
    parser.add_argument("--data-dir", type=Path, help="folder containing the four node directories")
    parser.add_argument("--output-dir", type=Path, help="output folder (default: results)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rounds", type=positive_int, default=200)
    parser.add_argument("--local-steps", type=positive_int, default=45)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = args.data_dir or root / "data"
    output_dir = args.output_dir or root / "results"
    if not data_dir.is_dir():
        parser.error(f"Data folder {data_dir} not found. Set --data-dir to the station data folder.")
    config = Config(rounds=args.rounds, local_steps=args.local_steps)
    torch.set_num_threads(4)
    clients = load_data(data_dir)
    station_ids = [str(client.station_id) for client in clients]
    capacities = [client.capacity_mw for client in clients]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training on {len(clients)} stations using {args.device}.", flush=True)
    def report(row):
        print(f'Validation, round {row["round"]:3d}  nRMSE {row["nrmse"]:.4f}%  nMAE {row["nmae"]:.4f}%', flush=True)

    result = fit(clients, config, args.device, on_evaluation=report)

    target_times = np.concatenate([c.timestamp_ns[c.test + 1] for c in clients])
    np.savez_compressed(
        output_dir / "predictions.npz",
        actual=result["actual"],
        prediction=result["prediction"],
        station=np.asarray(result["station"], dtype=str),
        target_time=target_times.astype("datetime64[ns]"),
        station_ids=np.asarray(station_ids),
        capacities_mw=np.asarray(capacities, dtype=np.float64),
    )
    write_metrics(result["metrics"], output_dir / "metrics.csv")
    torch.save({
        "vectors": result["vectors"],
        "station_ids": station_ids,
        "config": asdict(config),
        "round": result["round"],
        "input_protocol": "24h_power_nwp_and_calendar",
    }, output_dir / "checkpoint.pt")
    scores = result["metrics"]["average"]
    print(f'Test, final round {result["round"]}: nRMSE {scores["nrmse"]:.4f}%  nMAE {scores["nmae"]:.4f}%')
    print(f"Saved forecasts, metrics and models to {output_dir}")


if __name__ == "__main__":
    main()
