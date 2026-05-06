from __future__ import annotations

import hashlib
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import DataConfig
from .utils import FEET_TO_METERS


FEATURE_COLUMNS = [
    "local_x_m",
    "local_y_m",
    "velocity_mps",
    "acceleration_mps2",
    "lane_id",
    "vehicle_length_m",
    "vehicle_width_m",
    "space_headway_m",
    "time_headway_s",
    "delta_x_m",
    "delta_y_m",
    "preceding_exists",
    "following_exists",
]


CSV_COLUMNS = [
    "Vehicle_ID",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "v_length",
    "v_Width",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
    "Location",
]


@dataclass(slots=True)
class ClientData:
    client_id: int
    location: str
    train_x: np.ndarray
    train_y: np.ndarray
    eval_x: np.ndarray
    eval_y: np.ndarray
    comm_positions_m: np.ndarray
    comm_speed_mps: np.ndarray
    comm_acc_mps2: np.ndarray
    lane_ids: np.ndarray
    comm_space_headway_m: np.ndarray
    comm_time_headway_s: np.ndarray
    timestamps_s: np.ndarray
    class_balance: float

    @property
    def num_train_samples(self) -> int:
        return int(self.train_y.shape[0])

    @property
    def num_eval_samples(self) -> int:
        return int(self.eval_y.shape[0])


@dataclass(slots=True)
class RoadGeometry:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    roadside_x_m: float
    roadside_y_m: float
    center_x_m: float
    center_y_m: float


@dataclass(slots=True)
class FederatedDataset:
    train_clients: dict[int, ClientData]
    val_clients: dict[int, ClientData]
    test_clients: dict[int, ClientData]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    pos_weight: float
    road_geometry: dict[str, RoadGeometry]
    input_dim: int

    def all_train_client_ids(self) -> list[int]:
        return list(self.train_clients.keys())


class NGSIMDatasetBuilder:
    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build(self) -> FederatedDataset:
        cache_path = self.cache_dir / f"federated_{self._cache_key()}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                return pickle.load(handle)

        candidate_vehicle_ids = self._collect_candidate_vehicle_ids()
        df = self._load_filtered_dataframe(candidate_vehicle_ids)
        dataset = self._build_federated_bundle(df)

        with cache_path.open("wb") as handle:
            pickle.dump(dataset, handle)
        return dataset

    def _cache_key(self) -> str:
        payload = "|".join(
            [
                Path(self.config.dataset_path).name,
                ",".join(self.config.locations),
                str(self.config.max_vehicles),
                str(self.config.seq_len),
                str(self.config.horizon),
                str(self.config.min_frames_per_vehicle),
                self.config.dataset_version_tag,
            ]
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _collect_candidate_vehicle_ids(self) -> set[int]:
        counts: Counter[int] = Counter()
        for chunk in pd.read_csv(
            self.config.dataset_path,
            usecols=["Vehicle_ID", "Location"],
            chunksize=self.config.chunksize,
        ):
            mask = chunk["Location"].isin(self.config.locations)
            if not mask.any():
                continue
            counts.update(chunk.loc[mask, "Vehicle_ID"].astype(int).tolist())

        top_ids = [vehicle_id for vehicle_id, _ in counts.most_common(self.config.candidate_vehicle_pool)]
        return set(top_ids)

    def _load_filtered_dataframe(self, vehicle_ids: set[int]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            self.config.dataset_path,
            usecols=CSV_COLUMNS,
            chunksize=self.config.chunksize,
        ):
            mask = chunk["Location"].isin(self.config.locations) & chunk["Vehicle_ID"].isin(vehicle_ids)
            if mask.any():
                frames.append(chunk.loc[mask].copy())

        if not frames:
            raise RuntimeError("No rows matched the requested NGSIM configuration.")

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates()
        df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
        df["Global_Time"] = df["Global_Time"].astype(np.int64)
        df = df.sort_values(["Vehicle_ID", "Global_Time"]).reset_index(drop=True)
        return df

    def _build_federated_bundle(self, df: pd.DataFrame) -> FederatedDataset:
        client_records: list[
            tuple[int, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        road_geometry = self._compute_road_geometry(df)

        grouped = df.groupby("Vehicle_ID", sort=True)
        for vehicle_id, track in grouped:
            if len(track) < self.config.min_frames_per_vehicle:
                continue
            location = str(track["Location"].iloc[0])
            windows = self._vehicle_windows(track)
            if windows is None:
                continue
            client_records.append((vehicle_id, location, *windows))

        if not client_records:
            raise RuntimeError("The dataset did not produce any valid federated clients.")

        client_records = sorted(client_records, key=lambda item: item[2].shape[0], reverse=True)[: self.config.max_vehicles]
        rng = np.random.default_rng(self.config.random_seed)
        vehicle_ids = np.array([record[0] for record in client_records], dtype=int)
        rng.shuffle(vehicle_ids)

        n_total = len(vehicle_ids)
        n_train = max(1, int(n_total * self.config.vehicle_train_fraction))
        n_val = max(1, int(n_total * self.config.vehicle_val_fraction))
        train_ids = set(vehicle_ids[:n_train].tolist())
        val_ids = set(vehicle_ids[n_train : n_train + n_val].tolist())
        test_ids = set(vehicle_ids[n_train + n_val :].tolist())
        if not test_ids:
            test_ids = set(vehicle_ids[-max(1, n_total // 5) :].tolist())
            train_ids = set(vehicle_ids[: max(1, n_total - len(test_ids) - len(val_ids))].tolist())

        train_arrays = [record[2] for record in client_records if record[0] in train_ids]
        train_concat = np.concatenate(train_arrays, axis=0)
        feature_mean = train_concat.mean(axis=(0, 1))
        feature_std = train_concat.std(axis=(0, 1))
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

        train_labels = np.concatenate([record[3] for record in client_records if record[0] in train_ids], axis=0)
        positive = float(train_labels.sum())
        negative = float(train_labels.shape[0] - positive)
        pos_weight = negative / max(positive, 1.0)

        train_clients: dict[int, ClientData] = {}
        val_clients: dict[int, ClientData] = {}
        test_clients: dict[int, ClientData] = {}

        for vehicle_id, location, windows_x, windows_y, comm_positions, comm_state, comm_headway, timestamps in client_records:
            x_norm = (windows_x - feature_mean[None, None, :]) / feature_std[None, None, :]
            split_idx = max(1, int(len(windows_y) * self.config.window_train_fraction))
            train_x = x_norm[:split_idx]
            train_y = windows_y[:split_idx]
            eval_x = x_norm[split_idx:] if split_idx < len(windows_y) else x_norm[-1:]
            eval_y = windows_y[split_idx:] if split_idx < len(windows_y) else windows_y[-1:]

            client = ClientData(
                client_id=int(vehicle_id),
                location=location,
                train_x=train_x.astype(np.float32),
                train_y=train_y.astype(np.float32),
                eval_x=eval_x.astype(np.float32),
                eval_y=eval_y.astype(np.float32),
                comm_positions_m=comm_positions.astype(np.float32),
                comm_speed_mps=comm_state[:, 0].astype(np.float32),
                comm_acc_mps2=comm_state[:, 1].astype(np.float32),
                lane_ids=comm_state[:, 2].astype(np.float32),
                comm_space_headway_m=comm_headway[:, 0].astype(np.float32),
                comm_time_headway_s=comm_headway[:, 1].astype(np.float32),
                timestamps_s=timestamps.astype(np.float32),
                class_balance=float(windows_y.mean()),
            )

            if vehicle_id in train_ids:
                if client.num_train_samples >= self.config.min_windows_per_client:
                    train_clients[client.client_id] = client
            elif vehicle_id in val_ids:
                val_clients[client.client_id] = client
            else:
                test_clients[client.client_id] = client

        if not train_clients:
            raise RuntimeError("No training clients survived the minimum window threshold.")

        return FederatedDataset(
            train_clients=train_clients,
            val_clients=val_clients,
            test_clients=test_clients,
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
            pos_weight=float(pos_weight),
            road_geometry=road_geometry,
            input_dim=len(FEATURE_COLUMNS),
        )

    def _compute_road_geometry(self, df: pd.DataFrame) -> dict[str, RoadGeometry]:
        geometry: dict[str, RoadGeometry] = {}
        converted = df.copy()
        converted["Local_X"] = converted["Local_X"].astype(float) * FEET_TO_METERS
        converted["Local_Y"] = converted["Local_Y"].astype(float) * FEET_TO_METERS
        for location, group in converted.groupby("Location"):
            x_min = float(group["Local_X"].min())
            x_max = float(group["Local_X"].max())
            y_min = float(group["Local_Y"].min())
            y_max = float(group["Local_Y"].max())
            center_x = float(0.5 * (x_min + x_max))
            center_y = float(0.5 * (y_min + y_max))
            roadside_x = float(x_min - 6.0)
            roadside_y = center_y
            geometry[str(location)] = RoadGeometry(
                x_min_m=x_min,
                x_max_m=x_max,
                y_min_m=y_min,
                y_max_m=y_max,
                roadside_x_m=roadside_x,
                roadside_y_m=roadside_y,
                center_x_m=center_x,
                center_y_m=center_y,
            )
        return geometry

    def _vehicle_windows(
        self, track: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        track = track.copy()
        track = track.fillna({"Preceding": 0, "Following": 0, "Space_Headway": 0.0, "Time_Headway": 0.0})

        local_x = track["Local_X"].astype(float).to_numpy(dtype=np.float64) * FEET_TO_METERS
        local_y = track["Local_Y"].astype(float).to_numpy(dtype=np.float64) * FEET_TO_METERS
        velocity = track["v_Vel"].astype(float).to_numpy(dtype=np.float64) * FEET_TO_METERS
        acceleration = track["v_Acc"].astype(float).to_numpy(dtype=np.float64) * FEET_TO_METERS
        lane_id = track["Lane_ID"].astype(float).fillna(0.0).to_numpy(dtype=np.float64)
        vehicle_length = track["v_length"].astype(float).to_numpy(dtype=np.float64) * FEET_TO_METERS
        vehicle_width = track["v_Width"].astype(float).to_numpy(dtype=np.float64) * FEET_TO_METERS
        space_headway = track["Space_Headway"].astype(float).fillna(0.0).to_numpy(dtype=np.float64) * FEET_TO_METERS
        time_headway = track["Time_Headway"].astype(float).fillna(0.0).to_numpy(dtype=np.float64)
        preceding_exists = (track["Preceding"].astype(float).fillna(0.0).to_numpy(dtype=np.float64) > 0).astype(np.float64)
        following_exists = (track["Following"].astype(float).fillna(0.0).to_numpy(dtype=np.float64) > 0).astype(np.float64)
        timestamps = (track["Global_Time"].astype(np.int64).to_numpy(dtype=np.float64) - track["Global_Time"].iloc[0]) / 1000.0

        delta_x = np.diff(local_x, prepend=local_x[0])
        delta_y = np.diff(local_y, prepend=local_y[0])

        features = np.column_stack(
            [
                local_x,
                local_y,
                velocity,
                acceleration,
                lane_id,
                vehicle_length,
                vehicle_width,
                space_headway,
                time_headway,
                delta_x,
                delta_y,
                preceding_exists,
                following_exists,
            ]
        )

        seq_len = self.config.seq_len
        horizon = self.config.horizon
        max_start = len(track) - seq_len - horizon
        if max_start < 1:
            return None

        window_list: list[np.ndarray] = []
        label_list: list[int] = []
        comm_positions: list[np.ndarray] = []
        comm_state: list[np.ndarray] = []
        comm_headway: list[np.ndarray] = []
        comm_timestamps: list[float] = []

        for start_idx in range(max_start):
            current_slice = slice(start_idx, start_idx + seq_len)
            future_slice = slice(start_idx + seq_len, start_idx + seq_len + horizon)
            current_lane = lane_id[start_idx + seq_len - 1]
            future_lanes = lane_id[future_slice]
            lane_change_label = int(np.any(future_lanes != current_lane))
            window = features[current_slice]
            window_list.append(window)
            label_list.append(lane_change_label)
            snapshot_idx = start_idx + seq_len - 1
            comm_positions.append(np.array([local_x[snapshot_idx], local_y[snapshot_idx]], dtype=np.float64))
            comm_state.append(np.array([velocity[snapshot_idx], acceleration[snapshot_idx], lane_id[snapshot_idx]], dtype=np.float64))
            comm_headway.append(np.array([space_headway[snapshot_idx], time_headway[snapshot_idx]], dtype=np.float64))
            comm_timestamps.append(float(timestamps[snapshot_idx]))

        windows_x = np.stack(window_list, axis=0)
        windows_y = np.asarray(label_list, dtype=np.int64)

        if windows_y.shape[0] < self.config.min_windows_per_client:
            return None
        return (
            windows_x,
            windows_y,
            np.stack(comm_positions, axis=0),
            np.stack(comm_state, axis=0),
            np.stack(comm_headway, axis=0),
            np.asarray(comm_timestamps, dtype=np.float64),
        )


def iter_eval_clients(dataset: FederatedDataset) -> Iterable[ClientData]:
    for group in (dataset.val_clients, dataset.test_clients):
        for client in group.values():
            yield client
