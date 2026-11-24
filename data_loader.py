from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


LOOKBACK = 24
CHINA_STATIONS = {
    "15": (47, 76),
    "16": (92, 96, 105, 122, 131, 289),
    "26": (345, 347, 351, 355, 358),
    "31": (377, 552, 553),
}
CHINA_FIELDS = (
    "latitude", "longitude",
    "wind100_speed_ms", "wind100_direction_deg", "wind100_u_ms", "wind100_v_ms",
    "wind10_speed_ms", "wind10_direction_deg", "wind10_u_ms", "wind10_v_ms",
    "temperature_c", "pressure_kpa", "relative_humidity_pct",
)
CHINA_NWP = [f"nwp_{point}_{field}" for point in ("sw", "se", "nw", "ne") for field in CHINA_FIELDS]


@dataclass
class Client:
    station_id: str
    capacity_mw: float
    target_scale_mw: float
    timestamp_ns: np.ndarray
    features: np.ndarray
    power_mw: np.ndarray
    power_scaled: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def batch(self, origins, device):
        origins = np.asarray(origins)
        indices = origins[:, None] + np.arange(1 - LOOKBACK, 1)
        x = torch.from_numpy(self.features[indices]).to(device)
        y = torch.from_numpy(self.power_scaled[origins + 1, None]).to(device)
        return x, y


def _client(station_id, capacity_mw, time, weather, power):
    train_end = int(len(time) * 0.70)
    validation_end = int(len(time) * 0.85)
    train_power = power[:train_end]
    if train_power.size == 0:
        raise ValueError(f"No training power at station {station_id}")
    scale = float(np.quantile(train_power, 0.99))
    if scale < 1e-6:
        scale = max(float(train_power.std()), 1.0)
    scaled = power / scale
    day = 2.0 * np.pi * (time.hour.to_numpy() * 60 + time.minute.to_numpy()) / 1440.0
    year = 2.0 * np.pi * (time.dayofyear.to_numpy() - 1) / 365.25
    features = np.column_stack((
        weather, np.sin(day), np.cos(day), np.sin(year), np.cos(year), scaled,
    )).astype(np.float32)

    origins = np.arange(LOOKBACK - 1, len(time) - 1, dtype=np.int32)
    client = Client(
        station_id, capacity_mw, scale, time.as_unit("ns").asi8, features, power, scaled,
        origins[origins + 1 < train_end],
        origins[(origins + 1 >= train_end) & (origins + 1 < validation_end)],
        origins[origins + 1 >= validation_end],
    )
    if min(map(len, (client.train, client.validation, client.test))) == 0:
        raise ValueError(f"Insufficient complete windows at station {station_id}")
    return client


def load_data(data_dir):
    root = Path(data_dir)
    clients = []
    for node, station_ids in CHINA_STATIONS.items():
        found = set()
        for path in sorted((root / node).glob("1_*.csv")):
            _, station_id, capacity, _, _ = path.stem.split("_")
            if int(station_id) not in station_ids:
                continue
            if station_id in found:
                raise ValueError(f"Duplicate station {station_id} in {path.parent}")
            found.add(station_id)
            frame = pd.read_csv(path, usecols=["DATA_TIME", "REAL_POWER", *CHINA_NWP])
            time = pd.DatetimeIndex(pd.to_datetime(frame.pop("DATA_TIME")))
            if not (time.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all():
                raise ValueError(f"Expected consecutive 15-minute timestamps in {path}")
            frame.index = time
            frame = frame.apply(pd.to_numeric, errors="coerce")
            hourly_power = frame["REAL_POWER"].resample("h")
            if not (time[0] == time[0].floor("h") and hourly_power.size().eq(4).all()):
                raise ValueError(f"Incomplete hourly record in {path}")
            means = hourly_power.mean()
            power = np.maximum(means.to_numpy(dtype=np.float32), 0.0)
            if not np.isfinite(power).all():
                raise ValueError(f"Missing hourly power in {path}")
            weather = frame.loc[means.index, CHINA_NWP].to_numpy(dtype=np.float32)
            clients.append(_client(station_id, float(capacity), means.index, weather, power))
        absent = set(map(str, station_ids)) - found
        if absent:
            raise FileNotFoundError(f"Missing wind station(s) in {root / node}: {sorted(absent)}")

    parts = []
    for client in clients:
        end = int(len(client.power_mw) * 0.70)
        parts.append(client.features[:end])
    train = np.concatenate(parts)
    median = np.nanmedian(np.where(np.isfinite(train), train, np.nan), axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    rows, columns = np.where(~np.isfinite(train))
    train[rows, columns] = median[columns]
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    for client in clients:
        rows, columns = np.where(~np.isfinite(client.features))
        client.features[rows, columns] = median[columns]
        client.features -= mean
        client.features /= std
    return sorted(clients, key=lambda client: int(client.station_id))
