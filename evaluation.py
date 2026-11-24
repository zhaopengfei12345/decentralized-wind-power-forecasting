import argparse
import csv
from pathlib import Path

import numpy as np


def station_metrics(actual, prediction, station, capacities_mw):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    station = np.asarray(station).astype(str).reshape(-1)
    if not (actual.shape == prediction.shape == station.shape):
        raise ValueError("Each forecast must have one target and one station ID.")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Power values must be finite.")
    if not capacities_mw or set(station) != {str(key) for key in capacities_mw}:
        raise ValueError("Forecast station IDs and capacity records do not match.")

    rows = []
    for key, capacity in capacities_mw.items():
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError(f"Invalid rated capacity for station {key}.")
        mask = station == str(key)
        error = (prediction[mask] - actual[mask]) / capacity
        rows.append({
            "station": str(key),
            "nrmse": float(100 * np.sqrt(np.mean(error ** 2))),
            "nmae": float(100 * np.mean(np.abs(error))),
        })
    average = {name: float(np.mean([r[name] for r in rows])) for name in ("nrmse", "nmae")}
    return {"stations": rows, "average": average}


def write_metrics(metrics, path):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Station", "nRMSE (%)", "nMAE (%)"])
        for row in metrics["stations"]:
            writer.writerow([row["station"], row["nrmse"], row["nmae"]])
        writer.writerow(["Average", metrics["average"]["nrmse"], metrics["average"]["nmae"]])


def main():
    parser = argparse.ArgumentParser(description="Evaluate wind power forecasts.")
    parser.add_argument("predictions", type=Path, help="predictions.npz saved by main.py")
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        capacities = dict(zip(data["station_ids"].tolist(), data["capacities_mw"].tolist()))
        metrics = station_metrics(data["actual"], data["prediction"], data["station"], capacities)
    print(f'{"Station":>8} {"nRMSE (%)":>12} {"nMAE (%)":>12}')
    for row in [*metrics["stations"], {"station": "Average", **metrics["average"]}]:
        print(f'{row["station"]:>8} {row["nrmse"]:12.4f} {row["nmae"]:12.4f}')


if __name__ == "__main__":
    main()
