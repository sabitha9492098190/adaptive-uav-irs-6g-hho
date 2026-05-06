from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .config import CommunicationConfig, MobilityConfig
from .data import ClientData, RoadGeometry
from .utils import (
    SPEED_OF_LIGHT,
    bessel_j0,
    db_to_linear,
    dbm_to_watts,
    linear_to_db,
    noise_power_watts,
)


@dataclass(slots=True)
class ControlAction:
    uav_position_m: np.ndarray
    tx_power_dbm: float
    bandwidth_hz: float
    participation_ratio: float
    irs_phases_rad: np.ndarray
    irs_active: bool
    uav_mobile: bool
    round_idx: int = -1


@dataclass(slots=True)
class ClientSnapshot:
    position_m: np.ndarray
    speed_mps: float
    acceleration_mps2: float
    lane_id: float
    space_headway_m: float
    time_headway_s: float
    timestamp_s: float


@dataclass(slots=True)
class LinkMetrics:
    sinr_db: float
    rate_bps: float
    effective_rate_bps: float
    latency_s: float
    reliability: float
    throughput_bps: float
    ber: float
    outage: float
    tx_energy_j: float
    link_distance_m: float


class CommunicationEnvironment:
    def __init__(
        self,
        communication: CommunicationConfig,
        mobility: MobilityConfig,
        road_geometry: dict[str, RoadGeometry],
        seed: int,
    ) -> None:
        self.communication = communication
        self.mobility = mobility
        self.road_geometry = road_geometry
        self.seed = seed
        self.noise_power_w = noise_power_watts(communication.noise_psd_dbm_hz, communication.bandwidth_hz)

    def default_action(self, location: str, irs_active: bool, uav_mobile: bool) -> ControlAction:
        geometry = self.road_geometry[location]
        position = np.array(
            [geometry.center_x_m, geometry.center_y_m, self.mobility.uav_altitude_m],
            dtype=np.float64,
        )
        phases = np.zeros(self.communication.irs_elements, dtype=np.float64)
        return ControlAction(
            uav_position_m=position,
            tx_power_dbm=self.communication.tx_power_dbm,
            bandwidth_hz=self.communication.bandwidth_hz,
            participation_ratio=0.3,
            irs_phases_rad=phases,
            irs_active=irs_active,
            uav_mobile=uav_mobile,
        )

    def bounds(self, location: str) -> tuple[np.ndarray, np.ndarray]:
        geometry = self.road_geometry[location]
        lower = np.array(
            [
                geometry.x_min_m,
                geometry.y_min_m,
                self.mobility.uav_altitude_min_m,
                self.communication.tx_power_min_dbm,
                self.communication.bandwidth_min_hz,
                0.15,
                *([0.0] * self.communication.irs_elements),
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                geometry.x_max_m,
                geometry.y_max_m,
                self.mobility.uav_altitude_max_m,
                self.communication.tx_power_max_dbm,
                self.communication.bandwidth_max_hz,
                0.60,
                *([2.0 * math.pi] * self.communication.irs_elements),
            ],
            dtype=np.float64,
        )
        return lower, upper

    def vector_to_action(
        self,
        location: str,
        vector: np.ndarray,
        previous_action: ControlAction | None,
        irs_active: bool,
        uav_mobile: bool,
        round_idx: int,
    ) -> ControlAction:
        geometry = self.road_geometry[location]
        clipped = np.clip(vector, *self.bounds(location))
        position = np.array([clipped[0], clipped[1], clipped[2]], dtype=np.float64)
        if previous_action is not None and uav_mobile:
            max_distance = self.mobility.uav_max_speed_mps * self.mobility.round_duration_s
            delta = position - previous_action.uav_position_m
            distance = float(np.linalg.norm(delta))
            if distance > max_distance:
                position = previous_action.uav_position_m + delta / max(distance, 1e-9) * max_distance
        elif not uav_mobile and previous_action is not None:
            position = previous_action.uav_position_m.copy()
        elif not uav_mobile:
            position = np.array(
                [geometry.center_x_m, geometry.center_y_m, self.mobility.uav_altitude_m],
                dtype=np.float64,
            )

        phases = np.asarray(clipped[6:], dtype=np.float64)
        if previous_action is not None and irs_active:
            if round_idx - previous_action.round_idx <= self.communication.irs_update_cooldown_rounds:
                phases = previous_action.irs_phases_rad.copy()
        if not irs_active:
            phases = np.zeros_like(phases)

        action = ControlAction(
            uav_position_m=position,
            tx_power_dbm=float(clipped[3]),
            bandwidth_hz=float(clipped[4]),
            participation_ratio=float(clipped[5]),
            irs_phases_rad=self._quantize_phases(phases),
            irs_active=irs_active,
            uav_mobile=uav_mobile,
            round_idx=round_idx,
        )
        return action

    def heuristic_action(
        self,
        location: str,
        selected_snapshots: list[ClientSnapshot],
        previous_action: ControlAction | None,
        irs_active: bool,
        uav_mobile: bool,
        round_idx: int,
    ) -> ControlAction:
        action = self.default_action(location, irs_active=irs_active, uav_mobile=uav_mobile)
        if selected_snapshots and uav_mobile:
            positions = np.stack([snapshot.position_m for snapshot in selected_snapshots], axis=0)
            target_xy = positions.mean(axis=0)
            candidate_vector = np.concatenate(
                [
                    np.array(
                        [
                            target_xy[0],
                            target_xy[1],
                            self.mobility.uav_altitude_m,
                            self.communication.tx_power_dbm,
                            self.communication.bandwidth_hz,
                            0.30,
                        ],
                        dtype=np.float64,
                    ),
                    np.zeros(self.communication.irs_elements, dtype=np.float64),
                ]
            )
            action = self.vector_to_action(
                location=location,
                vector=candidate_vector,
                previous_action=previous_action,
                irs_active=irs_active,
                uav_mobile=uav_mobile,
                round_idx=round_idx,
            )
            if irs_active and selected_snapshots:
                action.irs_phases_rad = self._steer_phases(action, selected_snapshots[0], location)
        return action

    def snapshot_for_round(self, client: ClientData, round_idx: int) -> ClientSnapshot:
        stride = max(1, self.mobility.communication_stride_frames)
        idx = min(round_idx * stride, len(client.comm_positions_m) - 1)
        return ClientSnapshot(
            position_m=client.comm_positions_m[idx].astype(np.float64),
            speed_mps=float(client.comm_speed_mps[idx]),
            acceleration_mps2=float(client.comm_acc_mps2[idx]),
            lane_id=float(client.lane_ids[idx]),
            space_headway_m=float(client.comm_space_headway_m[idx]),
            time_headway_s=float(client.comm_time_headway_s[idx]),
            timestamp_s=float(client.timestamps_s[idx]),
        )

    def evaluate_link(
        self,
        action: ControlAction,
        snapshot: ClientSnapshot,
        client_id: int,
        location: str,
        model_size_bits: float,
        round_idx: int,
    ) -> LinkMetrics:
        tx_power_w = dbm_to_watts(action.tx_power_dbm)
        bandwidth_hz = action.bandwidth_hz
        noise_power_w = noise_power_watts(self.communication.noise_psd_dbm_hz, bandwidth_hz)
        effective_uav_position = self._drifted_uav_position(action, client_id, round_idx)
        blockage_probability = self._blockage_probability(snapshot)
        blockage_loss_linear = db_to_linear(-blockage_probability * self.communication.blockage_shadowing_db)

        direct_channel = self._direct_channel(
            uav_position_m=effective_uav_position,
            vehicle_position_m=snapshot.position_m,
            speed_mps=snapshot.speed_mps,
            client_id=client_id,
            round_idx=round_idx,
        )
        direct_channel *= math.sqrt(blockage_loss_linear)
        reflected_channel = 0.0j
        if action.irs_active:
            reflected_channel = self._irs_channel(
                uav_position_m=effective_uav_position,
                action=action,
                snapshot=snapshot,
                client_id=client_id,
                location=location,
                round_idx=round_idx,
            )
            reflected_channel *= math.sqrt(max(1.0 - 0.30 * blockage_probability, 0.10))

        combined = direct_channel + reflected_channel
        received_power_w = tx_power_w * (abs(combined) ** 2)
        distortion_w = self.communication.hardware_impairment_factor * received_power_w
        sinr_linear = received_power_w / max(noise_power_w + distortion_w, 1e-15)
        sinr_db = linear_to_db(sinr_linear)

        rate_bps = bandwidth_hz * math.log2(1.0 + sinr_linear)
        ber = 0.5 * math.erfc(math.sqrt(max(sinr_linear, 1e-12)))
        packet_success = 1.0 / (
            1.0 + math.exp(-self.communication.packet_success_steepness * (sinr_db - self.communication.sinr_outage_threshold_db))
        )
        reliability = max(0.0, min(packet_success * (1.0 - ber), 1.0))
        effective_rate_bps = rate_bps * max(reliability, 1e-6)
        tx_time_s = model_size_bits / max(effective_rate_bps, 1.0)
        irs_delay_s = self.communication.irs_reconfig_delay_s if action.irs_active else 0.0
        latency_s = tx_time_s + irs_delay_s + self.communication.aggregation_delay_s + self.communication.server_processing_delay_s
        tx_energy_j = tx_power_w * tx_time_s
        outage = float(sinr_db < self.communication.sinr_outage_threshold_db)
        horizontal = float(np.linalg.norm(effective_uav_position[:2] - snapshot.position_m))
        link_distance = float(np.linalg.norm(effective_uav_position - np.array([snapshot.position_m[0], snapshot.position_m[1], 0.0])))
        throughput_bps = model_size_bits / max(latency_s, 1e-6)

        return LinkMetrics(
            sinr_db=sinr_db,
            rate_bps=rate_bps,
            effective_rate_bps=effective_rate_bps,
            latency_s=latency_s,
            reliability=reliability,
            throughput_bps=throughput_bps,
            ber=ber,
            outage=outage,
            tx_energy_j=tx_energy_j,
            link_distance_m=max(link_distance, horizontal),
        )

    def _direct_channel(
        self,
        uav_position_m: np.ndarray,
        vehicle_position_m: np.ndarray,
        speed_mps: float,
        client_id: int,
        round_idx: int,
    ) -> complex:
        horizontal_distance = np.linalg.norm(uav_position_m[:2] - vehicle_position_m)
        distance = float(np.linalg.norm(uav_position_m - np.array([vehicle_position_m[0], vehicle_position_m[1], 0.0])))
        elevation_deg = math.degrees(math.atan2(uav_position_m[2], max(horizontal_distance, 1e-6)))
        p_los = 1.0 / (1.0 + self.communication.los_a * math.exp(-self.communication.los_b * (elevation_deg - self.communication.los_a)))
        path_loss_exp = (
            p_los * self.communication.los_path_loss_exponent
            + (1.0 - p_los) * self.communication.nlos_path_loss_exponent
        )
        shadowing_db = self._rng(client_id, round_idx, "shadow").normal(0.0, self.communication.shadowing_sigma_db)
        path_loss_db = self.communication.reference_path_loss_db + 10.0 * path_loss_exp * math.log10(max(distance, 1.0)) + shadowing_db
        amplitude = math.sqrt(10.0 ** (-path_loss_db / 10.0))

        wavelength = SPEED_OF_LIGHT / self.communication.carrier_frequency_hz
        arrival_angle = self._rng(client_id, round_idx, "doppler").uniform(0.0, math.pi)
        doppler_hz = (speed_mps / SPEED_OF_LIGHT) * self.communication.carrier_frequency_hz * math.cos(arrival_angle)
        rho = abs(bessel_j0(2.0 * math.pi * doppler_hz * self.communication.csi_delay_s))

        k = self.communication.rician_k_factor
        phase = 2.0 * math.pi * distance / wavelength
        los_component = complex(math.cos(phase), math.sin(phase))
        nlos_rng = self._rng(client_id, round_idx, "nlos")
        nlos_component = complex(
            nlos_rng.normal(0.0, 1.0) / math.sqrt(2.0),
            nlos_rng.normal(0.0, 1.0) / math.sqrt(2.0),
        )
        fading = math.sqrt(k / (k + 1.0)) * los_component + math.sqrt(1.0 / (k + 1.0)) * nlos_component
        innovation = complex(
            nlos_rng.normal(0.0, 1.0) / math.sqrt(2.0),
            nlos_rng.normal(0.0, 1.0) / math.sqrt(2.0),
        )
        aged_fading = rho * fading + math.sqrt(max(1.0 - rho**2, 0.0)) * innovation
        return amplitude * aged_fading

    def _irs_channel(
        self,
        uav_position_m: np.ndarray,
        action: ControlAction,
        snapshot: ClientSnapshot,
        client_id: int,
        location: str,
        round_idx: int,
    ) -> complex:
        geometry = self.road_geometry[location]
        irs_position = np.array([geometry.roadside_x_m, geometry.roadside_y_m, 6.0], dtype=np.float64)
        ui_distance = float(np.linalg.norm(uav_position_m - irs_position))
        iv_distance = float(np.linalg.norm(irs_position - np.array([snapshot.position_m[0], snapshot.position_m[1], 0.0])))
        wavelength = SPEED_OF_LIGHT / self.communication.carrier_frequency_hz

        ui_pathloss = db_to_linear(-(self.communication.reference_path_loss_db + 10.0 * self.communication.los_path_loss_exponent * math.log10(max(ui_distance, 1.0))))
        iv_pathloss = db_to_linear(-(self.communication.reference_path_loss_db + 10.0 * self.communication.nlos_path_loss_exponent * math.log10(max(iv_distance, 1.0))))

        rng = self._rng(client_id, round_idx, "irs")
        ui_phases = rng.uniform(0.0, 2.0 * math.pi, self.communication.irs_elements) + (2.0 * math.pi * ui_distance / wavelength)
        iv_phases = rng.uniform(0.0, 2.0 * math.pi, self.communication.irs_elements) + (2.0 * math.pi * iv_distance / wavelength)
        h_ui = np.sqrt(ui_pathloss) * np.exp(1j * ui_phases)
        h_iv = np.sqrt(iv_pathloss) * np.exp(1j * iv_phases)
        reflection = np.sum(h_ui * np.exp(1j * action.irs_phases_rad) * h_iv) / math.sqrt(self.communication.irs_elements)
        return self.communication.irs_gain * reflection

    def _steer_phases(
        self,
        action: ControlAction,
        snapshot: ClientSnapshot,
        location: str,
    ) -> np.ndarray:
        geometry = self.road_geometry[location]
        irs_position = np.array([geometry.roadside_x_m, geometry.roadside_y_m, 6.0], dtype=np.float64)
        ui_distance = float(np.linalg.norm(action.uav_position_m - irs_position))
        iv_distance = float(np.linalg.norm(irs_position - np.array([snapshot.position_m[0], snapshot.position_m[1], 0.0])))
        wavelength = SPEED_OF_LIGHT / self.communication.carrier_frequency_hz
        base_phase = -((2.0 * math.pi * ui_distance / wavelength) + (2.0 * math.pi * iv_distance / wavelength))
        phases = np.linspace(base_phase, base_phase + math.pi, self.communication.irs_elements, endpoint=False)
        return self._quantize_phases(phases)

    def _quantize_phases(self, phases: np.ndarray) -> np.ndarray:
        levels = 2 ** self.communication.irs_phase_bits
        quant_step = 2.0 * math.pi / levels
        phases = np.mod(phases, 2.0 * math.pi)
        return np.round(phases / quant_step) * quant_step

    def _blockage_probability(self, snapshot: ClientSnapshot) -> float:
        normalized_speed = min(snapshot.speed_mps / 35.0, 1.0)
        normalized_headway = 1.0 - min(snapshot.space_headway_m / 40.0, 1.0)
        normalized_time_headway = 1.0 - min(snapshot.time_headway_s / 3.0, 1.0)
        blockage = (
            self.communication.blockage_speed_weight * normalized_speed
            + self.communication.blockage_headway_weight * 0.5 * (normalized_headway + normalized_time_headway)
        )
        return float(np.clip(blockage, 0.0, 0.85))

    def _drifted_uav_position(self, action: ControlAction, client_id: int, round_idx: int) -> np.ndarray:
        if self.mobility.uav_drift_std_m <= 0.0:
            return action.uav_position_m
        rng = self._rng(client_id, round_idx, "drift")
        drift_xy = rng.normal(0.0, self.mobility.uav_drift_std_m, size=2)
        drift_z = rng.normal(0.0, self.mobility.uav_drift_std_m * 0.15)
        drift = np.array([drift_xy[0], drift_xy[1], drift_z], dtype=np.float64)
        return action.uav_position_m + drift

    def _rng(self, client_id: int, round_idx: int, tag: str) -> np.random.Generator:
        key = f"{self.seed}|{client_id}|{round_idx}|{tag}".encode("utf-8")
        seed = int(hashlib.md5(key).hexdigest()[:8], 16)
        return np.random.default_rng(seed)
