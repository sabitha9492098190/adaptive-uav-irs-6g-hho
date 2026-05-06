from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from .config import ModelConfig
from .data import ClientData
from .model import LocalModelUpdate


@dataclass(slots=True)
class EvaluationResult:
    loss: float
    accuracy: float
    precision: float
    recall: float


def _build_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_local_model(
    global_model: nn.Module,
    client: ClientData,
    model_config: ModelConfig,
    pos_weight: float,
    device: str,
    aggregation_weight: float,
) -> LocalModelUpdate:
    model = copy.deepcopy(global_model)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device)
    )

    train_loader = _build_loader(client.train_x, client.train_y, model_config.batch_size, shuffle=True)
    running_loss = []
    for _ in range(model_config.local_epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            clip_grad_norm_(model.parameters(), model_config.grad_clip_norm)
            optimizer.step()
            running_loss.append(float(loss.item()))

    eval_result = evaluate_model(model, [(client.eval_x, client.eval_y)], batch_size=model_config.batch_size, device=device)
    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return LocalModelUpdate(
        client_id=client.client_id,
        state_dict=state_dict,
        train_loss=float(np.mean(running_loss)) if running_loss else 0.0,
        eval_loss=eval_result.loss,
        eval_accuracy=eval_result.accuracy,
        num_samples=client.num_train_samples,
        aggregation_weight=float(aggregation_weight),
    )


def aggregate_fedavg(global_model: nn.Module, updates: list[LocalModelUpdate]) -> None:
    if not updates:
        return
    total_weight = sum(max(update.aggregation_weight, 1e-8) for update in updates)
    aggregated = {}
    for key in global_model.state_dict().keys():
        weighted_sum = None
        for update in updates:
            tensor = update.state_dict[key].float()
            weight = update.aggregation_weight / total_weight
            contribution = tensor * float(weight)
            weighted_sum = contribution if weighted_sum is None else weighted_sum + contribution
        aggregated[key] = weighted_sum
    global_model.load_state_dict(aggregated)


def evaluate_model(
    model: nn.Module,
    datasets: Iterable[tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    device: str,
) -> EvaluationResult:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    all_targets = []
    all_predictions = []
    losses = []

    with torch.no_grad():
        for data_x, data_y in datasets:
            loader = _build_loader(data_x.astype(np.float32), data_y.astype(np.float32), batch_size, shuffle=False)
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                losses.append(float(loss.item()))
                all_targets.extend(batch_y.cpu().numpy().tolist())
                all_predictions.extend(preds.cpu().numpy().tolist())

    targets = np.asarray(all_targets, dtype=np.float32)
    preds = np.asarray(all_predictions, dtype=np.float32)
    tp = float(np.sum((preds == 1) & (targets == 1)))
    fp = float(np.sum((preds == 1) & (targets == 0)))
    fn = float(np.sum((preds == 0) & (targets == 1)))
    accuracy = float(np.mean(preds == targets)) if targets.size else 0.0
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return EvaluationResult(
        loss=float(np.mean(losses)) if losses else 0.0,
        accuracy=accuracy,
        precision=float(precision),
        recall=float(recall),
    )
