from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector

from evaluation import station_metrics
from model import WindGRU


@dataclass
class Config:
    rounds: int = 200
    local_steps: int = 45
    batch_size: int = 256
    learning_rate: float = 0.095
    loss_mix: float = 0.2484879149384543
    weight_decay: float = 1.1973086498234576e-7
    gradient_clip: float = 5.0
    lambda_min: float = 0.032475597412840104
    lambda_max: float = 0.09742679223852034
    lambda_slope: float = 120.0
    distance_center: float = 0.015
    beta_min: float = 0.85
    beta_max: float = 0.985
    beta_scale: float = 12.0
    ema: float = 0.8
    probe_size: int = 128
    eval_every: int = 5
    eval_batch_size: int = 1024
    seed: int = 6101


def forecast_loss(prediction, target, weight):
    error = prediction - target
    return (weight * error.square() + (1 - weight) * error.abs()).mean()


@torch.no_grad()
def load_vector(model, vector):
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.copy_(vector[offset:offset + count].view_as(parameter))
        offset += count
    if offset != vector.numel():
        raise ValueError("The parameter vector does not match the GRU.")


def ring_matrix(size):
    if size < 2:
        raise ValueError("A ring needs at least two clients.")
    matrix = torch.zeros(size, size)
    for i in range(size):
        neighbors = [i, (i - 1) % size, (i + 1) % size]
        matrix[i, neighbors] = 1
    return matrix / matrix.sum(dim=1, keepdim=True)


@torch.inference_mode()
def probe_statistics(model, probes, vectors, config, device):
    model.eval()
    centroids, losses = [], []
    for (x, target), vector in zip(probes, vectors):
        load_vector(model, vector.to(device))
        representation = model.encode(x)
        prediction = model.head(representation)
        losses.append(forecast_loss(prediction, target, config.loss_mix).item())
        unit_vectors = F.normalize(representation, dim=1)
        centroid = F.normalize(unit_vectors.sum(dim=0) / len(x), dim=0)
        centroids.append(F.normalize(centroid, dim=0).cpu())
    return torch.stack(centroids), torch.tensor(losses)


def smooth(previous, current, alpha):
    if previous is None:
        return current
    return alpha * previous + (1 - alpha) * current


class AdaptiveControls:
    def __init__(self, mixing, config):
        self.config = config
        neighbors = mixing.clone()
        neighbors.fill_diagonal_(0)
        self.neighbors = neighbors / neighbors.sum(dim=1, keepdim=True)
        self.distance = None
        self.loss_change = None
        self.previous_loss = None

    def update(self, centroids, losses):
        cfg = self.config
        neighbor_centroids = F.normalize(self.neighbors @ centroids, dim=1)
        similarity = (F.normalize(centroids, dim=1) * neighbor_centroids).sum(dim=1)
        self.distance = smooth(self.distance, (1 - similarity).clamp(0, 2), cfg.ema)
        fraction = torch.sigmoid(cfg.lambda_slope * (cfg.distance_center - self.distance))
        lambdas = cfg.lambda_min + (cfg.lambda_max - cfg.lambda_min) * fraction

        if self.previous_loss is None:
            betas = torch.full_like(losses, cfg.beta_max)
        else:
            change = (losses - self.previous_loss).abs() / (self.previous_loss.abs() + 1e-12)
            self.loss_change = smooth(self.loss_change, change.clamp(0, 10), cfg.ema)
            stability = 2 * torch.sigmoid(-cfg.beta_scale * self.loss_change)
            betas = cfg.beta_min + (cfg.beta_max - cfg.beta_min) * stability
        self.previous_loss = losses
        return lambdas, betas


def local_update(model, client, anchor, strength, config, device, client_index, round_index):
    anchor = anchor.to(device)
    load_vector(model, anchor)
    parameters = list(model.parameters())
    anchor_parts = anchor.split([p.numel() for p in parameters])
    model.train()
    count = min(config.batch_size, len(client.train))
    for step in range(config.local_steps):
        seed = np.random.SeedSequence([config.seed, client_index, round_index, step])
        indexes = np.random.default_rng(seed).choice(len(client.train), count, replace=False)
        torch.manual_seed(int(seed.generate_state(1)[0]))
        x, target = client.batch(client.train[indexes], device)
        model.zero_grad(set_to_none=True)
        loss = forecast_loss(model(x), target, config.loss_mix)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at station {client.station_id}, round {round_index + 1}")
        loss.backward()
        if config.gradient_clip > 0:
            nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        with torch.no_grad():
            for parameter, reference in zip(parameters, anchor_parts):
                grad = parameter.grad + config.weight_decay * parameter
                grad = grad + strength * (parameter - reference.view_as(parameter))
                parameter.copy_(parameter - config.learning_rate * grad)
    return parameters_to_vector(parameters).detach().cpu()


@torch.inference_mode()
def predict(clients, vectors, config, device, split="test"):
    if split not in {"validation", "test"}:
        raise ValueError("Expected validation or test windows.")
    if len(clients) != len(vectors):
        raise ValueError("Expected one model vector per station.")
    model = WindGRU(clients[0].features.shape[1]).to(device)
    model.eval()
    actual, predicted, stations = [], [], []
    for client, vector in zip(clients, vectors):
        load_vector(model, vector.to(device))
        windows = getattr(client, split)
        for start in range(0, len(windows), config.eval_batch_size):
            origins = windows[start:start + config.eval_batch_size]
            x, _ = client.batch(origins, device)
            scaled = model(x).cpu().numpy()
            predicted.append(np.maximum(scaled, 0) * np.float32(client.target_scale_mw))
            actual.append(client.power_mw[origins + 1].reshape(-1, 1))
        stations.append(np.full(len(windows), str(client.station_id)))
    return {
        "actual": np.concatenate(actual),
        "prediction": np.concatenate(predicted),
        "station": np.concatenate(stations),
    }


def fit(clients, config, device, on_evaluation=None):
    if min(config.rounds, config.local_steps, config.probe_size, config.batch_size,
           config.eval_every, config.eval_batch_size) < 1:
        raise ValueError("Round, step, probe and batch counts must be positive.")
    if any(min(len(c.train), len(c.validation), len(c.test)) == 0 for c in clients):
        raise ValueError("Each station needs training, validation and test windows.")
    mixing = ring_matrix(len(clients))
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    model = WindGRU(clients[0].features.shape[1])
    initial = parameters_to_vector(model.parameters()).detach()
    current = initial.repeat(len(clients), 1)
    previous = current
    model = model.to(device)
    controls = AdaptiveControls(mixing, config)

    probes = []
    for client in clients:
        count = min(config.probe_size, len(client.train))
        indexes = np.linspace(0, len(client.train) - 1, count, dtype=np.int64)
        probes.append(client.batch(client.train[indexes], device))

    capacities = {c.station_id: c.capacity_mw for c in clients}
    history = []
    for round_index in range(config.rounds):
        centroids, losses = probe_statistics(model, probes, current, config, device)
        lambdas, betas = controls.update(centroids, losses)
        anchors = current + betas[:, None] * (current - previous)
        endpoints = [
            local_update(model, client, anchors[i], lambdas[i].item(), config, device, i, round_index)
            for i, client in enumerate(clients)
        ]
        previous, current = current, mixing @ torch.stack(endpoints)
        round_number = round_index + 1
        if round_number % config.eval_every == 0 or round_number == config.rounds:
            forecast = predict(clients, current, config, device, split="validation")
            metrics = station_metrics(**forecast, capacities_mw=capacities)
            average = metrics["average"]
            row = {"round": round_number, **average}
            history.append(row)
            if on_evaluation is not None:
                on_evaluation(row)
    forecast = predict(clients, current, config, device, split="test")
    metrics = station_metrics(**forecast, capacities_mw=capacities)
    return dict(vectors=current, round=config.rounds, history=history, metrics=metrics, **forecast)
