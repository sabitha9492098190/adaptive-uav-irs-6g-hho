from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DataConfig:
    dataset_path: str
    cache_dir: str = "artifacts/cache"
    locations: tuple[str, ...] = ("us-101",)
    candidate_vehicle_pool: int = 1000
    max_vehicles: int = 250
    min_frames_per_vehicle: int = 80
    min_windows_per_client: int = 8
    seq_len: int = 20
    horizon: int = 20
    frame_period_s: float = 0.1
    vehicle_train_fraction: float = 0.70
    vehicle_val_fraction: float = 0.10
    window_train_fraction: float = 0.80
    chunksize: int = 250_000
    random_seed: int = 42
    dataset_version_tag: str = "ngsim_v2"


@dataclass(slots=True)
class ModelConfig:
    input_dim: int = 13
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 5.0


@dataclass(slots=True)
class CommunicationConfig:
    carrier_frequency_hz: float = 28e9
    bandwidth_hz: float = 100e6
    bandwidth_min_hz: float = 60e6
    bandwidth_max_hz: float = 300e6
    noise_psd_dbm_hz: float = -174.0
    tx_power_dbm: float = 23.0
    tx_power_min_dbm: float = 18.0
    tx_power_max_dbm: float = 30.0
    los_path_loss_exponent: float = 2.2
    nlos_path_loss_exponent: float = 3.5
    reference_path_loss_db: float = 32.4
    shadowing_sigma_db: float = 4.0
    los_a: float = 9.61
    los_b: float = 0.16
    rician_k_factor: float = 6.0
    csi_delay_s: float = 0.1
    irs_elements: int = 16
    irs_phase_bits: int = 3
    irs_reconfig_delay_s: float = 0.010
    irs_gain: float = 1.0
    irs_update_cooldown_rounds: int = 1
    sinr_outage_threshold_db: float = 5.0
    target_model_size_bits: float = 2.5e6
    aggregation_delay_s: float = 0.015
    server_processing_delay_s: float = 0.010
    hardware_impairment_factor: float = 0.08
    packet_success_steepness: float = 1.7
    blockage_shadowing_db: float = 12.0
    blockage_speed_weight: float = 0.15
    blockage_headway_weight: float = 0.40


@dataclass(slots=True)
class MobilityConfig:
    uav_altitude_m: float = 120.0
    uav_altitude_min_m: float = 80.0
    uav_altitude_max_m: float = 150.0
    uav_max_speed_mps: float = 25.0
    uav_max_acc_mps2: float = 8.0
    uav_hover_power_w: float = 180.0
    uav_move_energy_per_m: float = 12.0
    uav_drift_std_m: float = 0.8
    roadside_offset_m: float = 6.0
    round_duration_s: float = 1.0
    communication_stride_frames: int = 10


@dataclass(slots=True)
class EnergyConfig:
    cycles_per_sample: float = 5e5
    energy_per_cycle_j: float = 1.2e-9
    client_energy_offset_j: float = 350.0
    client_energy_per_sample_j: float = 0.15
    max_client_energy_draw_j: float = 120.0


@dataclass(slots=True)
class HHOConfig:
    population_size: int = 14
    iterations: int = 20
    levy_beta: float = 1.5
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {
            "learning": 0.30,
            "rate": 0.20,
            "reliability": 0.20,
            "latency": 0.15,
            "energy": 0.10,
            "fairness": 0.05,
        }
    )


@dataclass(slots=True)
class ExperimentConfig:
    output_dir: str = "artifacts"
    metrics_filename: str = "metrics"
    num_rounds: int = 20
    clients_per_round: int = 12
    min_clients_per_round: int = 6
    cluster_count: int = 4
    participation_ratio_bounds: tuple[float, float] = (0.15, 0.60)
    baselines: tuple[str, ...] = ("cl_hho",)
    evaluate_every_round: bool = True
    device: str = "cpu"
    random_seed: int = 42
    verbose: bool = True


@dataclass(slots=True)
class Config:
    data: DataConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    hho: HHOConfig = field(default_factory=HHOConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def ensure_directories(self) -> None:
        Path(self.data.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.experiment.output_dir).mkdir(parents=True, exist_ok=True)
