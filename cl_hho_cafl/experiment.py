from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import torch
from sklearn.cluster import KMeans

from .channel import ClientSnapshot, CommunicationEnvironment, ControlAction, LinkMetrics
from .config import Config
from .data import ClientData, NGSIMDatasetBuilder, iter_eval_clients
from .fl import aggregate_fedavg, evaluate_model, train_local_model
from .hho import HarrisHawkOptimizer
from .model import GRULaneChangePredictor
from .utils import jains_fairness, set_global_seed


@dataclass(slots=True)
class ClientRuntimeState:
    energy_budget_j: float
    residual_energy_j: float
    recent_loss: float
    participation_count: int = 0


@dataclass(slots=True)
class SchemeProfile:
    name: str
    paper_label: str
    irs_active: bool
    uav_mobile: bool
    use_hho: bool
    communication_aware: bool
    use_clustering: bool
    random_selection: bool


@dataclass(slots=True)
class ScoredClient:
    client: ClientData
    snapshot: ClientSnapshot
    link_metrics: LinkMetrics
    utility: float
    compute_energy_j: float
    residual_energy_ratio: float
    cluster_vector: np.ndarray


class CLHHOExperiment:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.config.ensure_directories()
        set_global_seed(config.experiment.random_seed)
        self.dataset = NGSIMDatasetBuilder(config.data).build()
        self.env = CommunicationEnvironment(
            communication=config.communication,
            mobility=config.mobility,
            road_geometry=self.dataset.road_geometry,
            seed=config.experiment.random_seed,
        )
        self.location = config.data.locations[0]
        self.device = torch.device(config.experiment.device)
        self.scheme_profiles = self._build_scheme_profiles()

    def run(self) -> dict[str, dict]:
        self._log_section("System Initialization and Scenario Definition")
        self._log(
            f"Dataset path: {self.config.data.dataset_path}\n"
            f"Train clients: {len(self.dataset.train_clients)} | "
            f"Validation clients: {len(self.dataset.val_clients)} | "
            f"Test clients: {len(self.dataset.test_clients)}\n"
            f"Sequence length: {self.config.data.seq_len} | Prediction horizon: {self.config.data.horizon}\n"
            f"Rounds: {self.config.experiment.num_rounds} | Clients per round: {self.config.experiment.clients_per_round}"
        )
        self._log_section("Multi-Layer Channel Modeling")
        self._log(
            f"Carrier frequency: {self.config.communication.carrier_frequency_hz / 1e9:.1f} GHz | "
            f"Bandwidth: {self.config.communication.bandwidth_hz / 1e6:.1f} MHz\n"
            f"IRS elements: {self.config.communication.irs_elements} | "
            f"IRS phase bits: {self.config.communication.irs_phase_bits} | "
            f"IRS delay: {self.config.communication.irs_reconfig_delay_s * 1000.0:.1f} ms\n"
            f"UAV altitude range: {self.config.mobility.uav_altitude_min_m:.0f}-{self.config.mobility.uav_altitude_max_m:.0f} m | "
            f"UAV drift std: {self.config.mobility.uav_drift_std_m:.2f} m"
        )
        self._log_section("Cross-Layer Problem Formulation")
        self._log(
            f"Objective weights: {self.config.hho.objective_weights}\n"
            f"Schemes: {', '.join(self.config.experiment.baselines)}"
        )
        self._log_section("Adaptive Optimization Using Harris Hawk Optimization")
        self._log(
            f"HHO population size: {self.config.hho.population_size} | "
            f"HHO iterations: {self.config.hho.iterations}"
        )
        self._log_section("Communication-Aware Federated Learning Integration")
        self._log(
            f"Lane-change prediction task over NGSIM trajectories | "
            f"GRU hidden dimension: {self.config.model.hidden_dim} | "
            f"Local epochs: {self.config.model.local_epochs}"
        )

        base_model = self._create_model()
        base_state_dict = copy.deepcopy(base_model.state_dict())
        all_results: dict[str, dict] = {}

        for scheme in self.config.experiment.baselines:
            profile = self.scheme_profiles[scheme]
            self._log_section(f"Performance Evaluation")
            model = self._create_model()
            model.load_state_dict(base_state_dict)
            model.to(self.device)
            scheme_results = self._run_scheme(profile, model)
            all_results[scheme] = scheme_results
            self._log(self._render_scheme_line(profile.paper_label, scheme_results))

        metrics_path = Path(self.config.experiment.output_dir) / self.config.experiment.metrics_filename
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cl_hho = all_results.get("cl_hho", {})
        lines = [
            "CL-HHO METRICS REPORT",
            f"Scheme: {cl_hho.get('paper_label', 'CL-HHO')}",
            f"Average SINR (dB): {cl_hho.get('final_sinr_db', 0.0):.4f}",
            f"BER @ configured transmit power: {cl_hho.get('final_ber', 0.0):.6e}",
            f"Outage Probability: {cl_hho.get('final_outage_probability', 0.0):.4f}",
            f"End-to-End Latency (ms): {cl_hho.get('final_latency_ms', 0.0):.4f}",
            f"Throughput (Mbps): {cl_hho.get('final_throughput_mbps', 0.0):.4f}",
            f"Packet Delivery Ratio (%): {cl_hho.get('final_pdr', 0.0):.4f}",
            f"Final Model Accuracy: {cl_hho.get('final_accuracy', 0.0):.4f}",
            f"Convergence Rounds: {cl_hho.get('convergence_rounds', 0)}",
            f"Fairness Index (Jain): {cl_hho.get('final_fairness', 0.0):.4f}",
            f"Avg. Energy per FL Round (J): {cl_hho.get('average_total_round_energy_j', 0.0):.4f}",
            f"Energy Efficiency (Accuracy/J): {cl_hho.get('energy_efficiency_accuracy_per_j', 0.0):.6e}",
        ]
        self._log_section("Metrics File")
        
        if metrics_path.exists():
            print(metrics_path.read_text(encoding="utf-8"))
        else:
            print("Metrics file not found. No new results were saved.")
        return all_results

    def _run_scheme(self, profile: SchemeProfile, model: torch.nn.Module) -> dict:
        train_clients = list(self.dataset.train_clients.values())
        runtime_state = self._initial_runtime_state(train_clients)
        previous_action = self.env.default_action(
            self.location,
            irs_active=profile.irs_active,
            uav_mobile=profile.uav_mobile,
        )
        round_metrics: list[dict] = []
        eval_sets = [(client.eval_x, client.eval_y) for client in iter_eval_clients(self.dataset)]
        if not eval_sets:
            eval_sets = [(client.eval_x, client.eval_y) for client in train_clients]

        for round_idx in range(self.config.experiment.num_rounds):
            action = self._choose_action(profile, runtime_state, previous_action, round_idx)
            selected_clients, scored = self._select_clients(profile, action, runtime_state, round_idx)

            local_updates = []
            selected_link_metrics: list[LinkMetrics] = []
            total_client_energy_j = 0.0
            for scored_client in selected_clients:
                client = scored_client.client
                metrics = scored_client.link_metrics
                total_round_energy = scored_client.compute_energy_j + metrics.tx_energy_j
                state = runtime_state[client.client_id]
                if state.residual_energy_j < total_round_energy:
                    continue

                aggregation_weight = client.num_train_samples * max(metrics.reliability, 1e-3)
                update = train_local_model(
                    global_model=model,
                    client=client,
                    model_config=self.config.model,
                    pos_weight=self.dataset.pos_weight,
                    device=str(self.device),
                    aggregation_weight=aggregation_weight,
                )
                local_updates.append(update)
                selected_link_metrics.append(metrics)
                total_client_energy_j += total_round_energy
                state.recent_loss = update.eval_loss
                state.participation_count += 1
                state.residual_energy_j = max(0.0, state.residual_energy_j - total_round_energy)

            if local_updates:
                aggregate_fedavg(model, local_updates)

            eval_result = evaluate_model(
                model=model,
                datasets=eval_sets,
                batch_size=self.config.model.batch_size,
                device=str(self.device),
            )

            avg_sinr = float(np.mean([metrics.sinr_db for metrics in selected_link_metrics])) if selected_link_metrics else 0.0
            avg_latency_ms = (
                float(np.mean([metrics.latency_s for metrics in selected_link_metrics])) * 1000.0
                if selected_link_metrics
                else 0.0
            )
            avg_reliability = float(np.mean([metrics.reliability for metrics in selected_link_metrics])) if selected_link_metrics else 0.0
            avg_throughput_mbps = (
                float(np.mean([metrics.throughput_bps for metrics in selected_link_metrics])) / 1e6
                if selected_link_metrics
                else 0.0
            )
            avg_ber = float(np.mean([metrics.ber for metrics in selected_link_metrics])) if selected_link_metrics else 0.0
            outage_probability = (
                float(np.mean([metrics.outage for metrics in selected_link_metrics])) if selected_link_metrics else 0.0
            )
            fairness = jains_fairness([state.participation_count for state in runtime_state.values()])
            uav_round_energy = self._uav_energy(previous_action, action)
            round_metrics.append(
                {
                    "round": round_idx + 1,
                    "selected_clients": [item.client.client_id for item in selected_clients],
                    "selected_count": len(selected_clients),
                    "accuracy": eval_result.accuracy,
                    "loss": eval_result.loss,
                    "precision": eval_result.precision,
                    "recall": eval_result.recall,
                    "avg_sinr_db": avg_sinr,
                    "avg_latency_ms": avg_latency_ms,
                    "avg_reliability": avg_reliability,
                    "avg_throughput_mbps": avg_throughput_mbps,
                    "avg_ber": avg_ber,
                    "outage_probability": outage_probability,
                    "fairness": fairness,
                    "client_round_energy_j": total_client_energy_j,
                    "uav_round_energy_j": uav_round_energy,
                    "total_round_energy_j": total_client_energy_j + uav_round_energy,
                }
            )
            previous_action = action
            self._log_round(profile.paper_label, round_metrics[-1])

        convergence_round = self._convergence_round(round_metrics)
        avg_total_round_energy = float(np.mean([item["total_round_energy_j"] for item in round_metrics])) if round_metrics else 0.0
        final_accuracy = round_metrics[-1]["accuracy"] if round_metrics else 0.0
        return {
            "scheme": profile.name,
            "paper_label": profile.paper_label,
            "rounds": round_metrics,
            "final_accuracy": final_accuracy,
            "final_latency_ms": round_metrics[-1]["avg_latency_ms"] if round_metrics else 0.0,
            "final_sinr_db": round_metrics[-1]["avg_sinr_db"] if round_metrics else 0.0,
            "final_pdr": round_metrics[-1]["avg_reliability"] * 100.0 if round_metrics else 0.0,
            "final_throughput_mbps": round_metrics[-1]["avg_throughput_mbps"] if round_metrics else 0.0,
            "final_ber": round_metrics[-1]["avg_ber"] if round_metrics else 0.0,
            "final_outage_probability": round_metrics[-1]["outage_probability"] if round_metrics else 0.0,
            "final_fairness": round_metrics[-1]["fairness"] if round_metrics else 0.0,
            "convergence_rounds": convergence_round,
            "average_total_round_energy_j": avg_total_round_energy,
            "energy_efficiency_accuracy_per_j": final_accuracy / max(avg_total_round_energy, 1e-9),
        }

    def _choose_action(
        self,
        profile: SchemeProfile,
        runtime_state: dict[int, ClientRuntimeState],
        previous_action: ControlAction,
        round_idx: int,
    ) -> ControlAction:
        objective_clients = self._objective_candidate_clients(runtime_state)
        representative_snapshots = [self.env.snapshot_for_round(client, round_idx) for client in objective_clients]

        if not profile.use_hho:
            if not profile.irs_active and not profile.uav_mobile:
                return self.env.default_action(self.location, irs_active=False, uav_mobile=False)
            return self.env.heuristic_action(
                location=self.location,
                selected_snapshots=representative_snapshots,
                previous_action=previous_action,
                irs_active=profile.irs_active,
                uav_mobile=profile.uav_mobile,
                round_idx=round_idx,
            )

        lower, upper = self.env.bounds(self.location)
        objective = self._objective_factory(profile, runtime_state, previous_action, round_idx, objective_clients)
        optimizer = HarrisHawkOptimizer(
            lower_bounds=lower,
            upper_bounds=upper,
            population_size=self.config.hho.population_size,
            iterations=self.config.hho.iterations,
            beta=self.config.hho.levy_beta,
            seed=self.config.experiment.random_seed + round_idx,
        )
        result = optimizer.optimize(objective)
        return self.env.vector_to_action(
            location=self.location,
            vector=result.best_vector,
            previous_action=previous_action,
            irs_active=profile.irs_active,
            uav_mobile=profile.uav_mobile,
            round_idx=round_idx,
        )

    def _objective_factory(
        self,
        profile: SchemeProfile,
        runtime_state: dict[int, ClientRuntimeState],
        previous_action: ControlAction,
        round_idx: int,
        candidate_clients: list[ClientData],
    ):
        weights = self.config.hho.objective_weights

        def objective(vector: np.ndarray) -> float:
            action = self.env.vector_to_action(
                location=self.location,
                vector=vector,
                previous_action=previous_action,
                irs_active=profile.irs_active,
                uav_mobile=profile.uav_mobile,
                round_idx=round_idx,
            )
            scored = self._score_clients(action, runtime_state, round_idx, candidate_clients)
            selected = self._top_clients_from_scores(profile, scored, action.participation_ratio, round_idx)
            if not selected:
                return 1e9

            metrics = [item.link_metrics for item in selected]
            utilities = [item.utility for item in selected]
            compute_energy = [item.compute_energy_j + item.link_metrics.tx_energy_j for item in selected]
            fairness_after = 1.0 - jains_fairness(
                [
                    runtime_state[client_id].participation_count
                    + (1 if client_id in {item.client.client_id for item in selected} else 0)
                    for client_id in runtime_state
                ]
            )

            learning_term = float(np.mean(utilities))
            rate_term = float(np.mean([min(metric.effective_rate_bps / 8e7, 1.5) for metric in metrics]))
            reliability_term = float(np.mean([metric.reliability for metric in metrics]))
            latency_term = float(np.mean([metric.latency_s for metric in metrics])) / 0.30
            energy_term = float(np.mean(compute_energy)) / max(self.config.energy.max_client_energy_draw_j, 1e-6)
            return float(
                weights["latency"] * latency_term
                + weights["energy"] * energy_term
                + weights["fairness"] * fairness_after
                - weights["learning"] * learning_term
                - weights["rate"] * rate_term
                - weights["reliability"] * reliability_term
            )

        return objective

    def _select_clients(
        self,
        profile: SchemeProfile,
        action: ControlAction,
        runtime_state: dict[int, ClientRuntimeState],
        round_idx: int,
    ) -> tuple[list[ScoredClient], dict[int, ScoredClient]]:
        scored = self._score_clients(action, runtime_state, round_idx, list(self.dataset.train_clients.values()))
        selected = self._top_clients_from_scores(profile, scored, action.participation_ratio, round_idx)
        return selected, scored

    def _score_clients(
        self,
        action: ControlAction,
        runtime_state: dict[int, ClientRuntimeState],
        round_idx: int,
        clients: list[ClientData],
    ) -> dict[int, ScoredClient]:
        if not clients:
            return {}
        max_train_samples = max(client.num_train_samples for client in clients)
        scored: dict[int, ScoredClient] = {}

        for client in clients:
            snapshot = self.env.snapshot_for_round(client, round_idx)
            metrics = self.env.evaluate_link(
                action=action,
                snapshot=snapshot,
                client_id=client.client_id,
                location=self.location,
                model_size_bits=self.config.communication.target_model_size_bits,
                round_idx=round_idx,
            )
            compute_energy = self._client_compute_energy(client)
            state = runtime_state[client.client_id]
            if state.residual_energy_j < compute_energy + metrics.tx_energy_j:
                continue

            data_term = math.log1p(client.num_train_samples) / math.log1p(max_train_samples)
            loss_term = min(state.recent_loss / 1.5, 1.0)
            residual_energy_ratio = min(state.residual_energy_j / max(state.energy_budget_j, 1e-6), 1.0)
            fairness_term = 1.0 / (1.0 + state.participation_count)
            rate_term = min(metrics.effective_rate_bps / 8e7, 1.0)
            latency_term = max(0.0, 1.0 - metrics.latency_s / 0.25)
            mobility_term = min(snapshot.speed_mps / 35.0, 1.0)

            utility = (
                0.22 * metrics.reliability
                + 0.18 * rate_term
                + 0.13 * latency_term
                + 0.14 * loss_term
                + 0.10 * data_term
                + 0.09 * residual_energy_ratio
                + 0.07 * fairness_term
                + 0.07 * (1.0 - mobility_term)
            )
            cluster_vector = np.array(
                [
                    metrics.reliability,
                    min(metrics.effective_rate_bps / 8e7, 1.0),
                    min(metrics.latency_s / 0.25, 1.0),
                    residual_energy_ratio,
                    min(snapshot.speed_mps / 35.0, 1.0),
                    min(snapshot.time_headway_s / 3.0, 1.0),
                    min(snapshot.space_headway_m / 40.0, 1.0),
                    min(state.recent_loss / 1.5, 1.0),
                ],
                dtype=np.float64,
            )
            scored[client.client_id] = ScoredClient(
                client=client,
                snapshot=snapshot,
                link_metrics=metrics,
                utility=float(utility),
                compute_energy_j=float(compute_energy),
                residual_energy_ratio=float(residual_energy_ratio),
                cluster_vector=cluster_vector,
            )
        return scored

    def _top_clients_from_scores(
        self,
        profile: SchemeProfile,
        scored: dict[int, ScoredClient],
        participation_ratio: float,
        round_idx: int,
    ) -> list[ScoredClient]:
        ranked = sorted(scored.values(), key=lambda item: item.utility, reverse=True)
        if not ranked:
            return []

        max_by_ratio = max(
            self.config.experiment.min_clients_per_round,
            int(len(self.dataset.train_clients) * participation_ratio),
        )
        num_clients = min(self.config.experiment.clients_per_round, max_by_ratio, len(ranked))
        if num_clients <= 0:
            return []

        if profile.random_selection:
            rng = np.random.default_rng(self.config.experiment.random_seed + round_idx)
            indices = rng.permutation(len(ranked))[:num_clients]
            return [ranked[int(idx)] for idx in indices]

        if not profile.communication_aware:
            reliability_ranked = sorted(ranked, key=lambda item: item.link_metrics.reliability, reverse=True)
            return reliability_ranked[:num_clients]

        if not profile.use_clustering or len(ranked) <= num_clients:
            return ranked[:num_clients]

        n_clusters = min(self.config.experiment.cluster_count, num_clients, len(ranked))
        matrix = np.stack([item.cluster_vector for item in ranked], axis=0)
        model = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=self.config.experiment.random_seed + round_idx,
        )
        labels = model.fit_predict(matrix)
        cluster_best: list[ScoredClient] = []
        used_ids: set[int] = set()
        for cluster_idx in range(n_clusters):
            members = [ranked[i] for i, label in enumerate(labels) if label == cluster_idx]
            if not members:
                continue
            best = max(members, key=lambda item: item.utility)
            cluster_best.append(best)
            used_ids.add(best.client.client_id)

        remaining = [item for item in ranked if item.client.client_id not in used_ids]
        selected = cluster_best[:num_clients]
        if len(selected) < num_clients:
            selected.extend(remaining[: num_clients - len(selected)])
        return selected[:num_clients]

    def _objective_candidate_clients(self, runtime_state: dict[int, ClientRuntimeState]) -> list[ClientData]:
        clients = list(self.dataset.train_clients.values())
        ranked = sorted(
            clients,
            key=lambda client: (
                runtime_state[client.client_id].recent_loss,
                -runtime_state[client.client_id].participation_count,
                runtime_state[client.client_id].residual_energy_j,
            ),
            reverse=True,
        )
        return ranked[: min(len(ranked), 32)]

    def _initial_runtime_state(self, train_clients: list[ClientData]) -> dict[int, ClientRuntimeState]:
        state = {}
        for client in train_clients:
            budget = self.config.energy.client_energy_offset_j + self.config.energy.client_energy_per_sample_j * client.num_train_samples
            state[client.client_id] = ClientRuntimeState(
                energy_budget_j=float(budget),
                residual_energy_j=float(budget),
                recent_loss=0.69,
            )
        return state

    def _client_compute_energy(self, client: ClientData) -> float:
        return (
            self.config.energy.cycles_per_sample
            * client.num_train_samples
            * self.config.model.local_epochs
            * self.config.energy.energy_per_cycle_j
        )

    def _uav_energy(self, previous_action: ControlAction, current_action: ControlAction) -> float:
        move_distance = float(np.linalg.norm(current_action.uav_position_m - previous_action.uav_position_m))
        return (
            self.config.mobility.uav_hover_power_w * self.config.mobility.round_duration_s
            + self.config.mobility.uav_move_energy_per_m * move_distance
        )

    def _create_model(self) -> torch.nn.Module:
        return GRULaneChangePredictor(self.dataset.input_dim, self.config.model)

    def _convergence_round(self, rounds: list[dict]) -> int:
        if not rounds:
            return 0
        final_accuracy = rounds[-1]["accuracy"]
        threshold = 0.95 * final_accuracy
        for item in rounds:
            if item["accuracy"] >= threshold:
                return int(item["round"])
        return int(rounds[-1]["round"])

    def _build_scheme_profiles(self) -> dict[str, SchemeProfile]:
        return {
            "cl_hho": SchemeProfile(
                name="cl_hho",
                paper_label="CL-HHO",
                irs_active=True,
                uav_mobile=True,
                use_hho=True,
                communication_aware=True,
                use_clustering=True,
                random_selection=False,
            ),
        }

    def _render_scheme_line(self, label: str, summary: dict) -> str:
        return (
            f"{label}: accuracy={summary['final_accuracy']:.4f}, "
            f"latency_ms={summary['final_latency_ms']:.2f}, "
            f"sinr_db={summary['final_sinr_db']:.2f}, "
            f"pdr={summary['final_pdr']:.2f}, "
            f"energy_j={summary['average_total_round_energy_j']:.2f}"
        )

    def _log_round(self, label: str, round_summary: dict) -> None:
        self._log(
            f"{label} | Round {round_summary['round']:02d}/{self.config.experiment.num_rounds} | "
            f"clients={round_summary['selected_count']} | "
            f"accuracy={round_summary['accuracy']:.4f} | "
            f"latency_ms={round_summary['avg_latency_ms']:.2f} | "
            f"sinr_db={round_summary['avg_sinr_db']:.2f} | "
            f"pdr={round_summary['avg_reliability'] * 100.0:.2f}"
        )

    def _log_section(self, title: str) -> None:
        if self.config.experiment.verbose:
            print(f"\n=== {title} ===")

    def _log(self, message: str) -> None:
        if self.config.experiment.verbose:
            print(message)
