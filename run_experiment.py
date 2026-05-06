from __future__ import annotations

import argparse
from pathlib import Path

from cl_hho_cafl import CLHHOExperiment, Config
from cl_hho_cafl.config import DataConfig, ExperimentConfig, HHOConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full CL-HHO-CAFL simulation on NGSIM traces.")
    parser.add_argument("--dataset", help="Optional explicit path to the NGSIM CSV file.")
    parser.add_argument("--location", default="us-101", help="NGSIM location to use, e.g. us-101 or i-80.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for the plain-text metrics file.")
    parser.add_argument("--cache-dir", default="artifacts/cache", help="Directory for processed dataset cache.")
    parser.add_argument("--max-vehicles", type=int, default=250, help="Number of vehicle clients to retain.")
    parser.add_argument("--candidate-vehicle-pool", type=int, default=1000, help="Vehicle pool before filtering.")
    parser.add_argument("--rounds", type=int, default=20, help="Federated rounds per scheme.")
    parser.add_argument("--clients-per-round", type=int, default=12, help="Upper bound on selected clients.")
    parser.add_argument("--population-size", type=int, default=14, help="HHO population size.")
    parser.add_argument("--iterations", type=int, default=20, help="HHO iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate step logging.")
    parser.add_argument("--smoke", action="store_true", help="Run a smaller sanity-check configuration.")
    return parser


def detect_dataset_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).resolve()

    dataset_dir = Path.cwd() / "Dataset"
    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV dataset was found in the Dataset folder. "
            "Place the NGSIM CSV in ./Dataset or pass --dataset."
        )
    if len(csv_files) > 1:
        raise FileExistsError(
            "Multiple CSV files were found in the Dataset folder. "
            "Pass --dataset to choose one explicitly."
        )
    return csv_files[0].resolve()


def main() -> None:
    args = build_parser().parse_args()
    dataset_path = detect_dataset_path(args.dataset)

    data_cfg = DataConfig(
        dataset_path=str(dataset_path),
        cache_dir=args.cache_dir,
        locations=(args.location,),
        candidate_vehicle_pool=args.candidate_vehicle_pool,
        max_vehicles=args.max_vehicles,
        random_seed=args.seed,
    )
    experiment_cfg = ExperimentConfig(
        output_dir=args.output_dir,
        num_rounds=args.rounds,
        clients_per_round=args.clients_per_round,
        random_seed=args.seed,
        verbose=not args.quiet,
    )
    hho_cfg = HHOConfig(
        population_size=args.population_size,
        iterations=args.iterations,
    )

    if args.smoke:
        data_cfg.max_vehicles = min(data_cfg.max_vehicles, 60)
        data_cfg.candidate_vehicle_pool = min(data_cfg.candidate_vehicle_pool, 200)
        experiment_cfg.num_rounds = min(experiment_cfg.num_rounds, 4)
        experiment_cfg.clients_per_round = min(experiment_cfg.clients_per_round, 6)
        experiment_cfg.cluster_count = min(experiment_cfg.cluster_count, 3)
        hho_cfg.population_size = min(hho_cfg.population_size, 6)
        hho_cfg.iterations = min(hho_cfg.iterations, 4)

    config = Config(data=data_cfg, experiment=experiment_cfg, hho=hho_cfg)
    experiment = CLHHOExperiment(config)
    experiment.run()
    import Graph


if __name__ == "__main__":
    main()
