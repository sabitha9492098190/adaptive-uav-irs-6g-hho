from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(slots=True)
class HHOResult:
    best_vector: np.ndarray
    best_fitness: float
    fitness_trace: list[float]


class HarrisHawkOptimizer:
    def __init__(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        population_size: int,
        iterations: int,
        beta: float,
        seed: int,
    ) -> None:
        self.lower_bounds = lower_bounds.astype(np.float64)
        self.upper_bounds = upper_bounds.astype(np.float64)
        self.population_size = population_size
        self.iterations = iterations
        self.beta = beta
        self.rng = np.random.default_rng(seed)

    def optimize(self, objective: Callable[[np.ndarray], float]) -> HHOResult:
        dim = self.lower_bounds.shape[0]
        hawks = self.rng.uniform(self.lower_bounds, self.upper_bounds, size=(self.population_size, dim))
        fitness = np.asarray([objective(candidate) for candidate in hawks], dtype=np.float64)
        best_idx = int(np.argmin(fitness))
        rabbit = hawks[best_idx].copy()
        rabbit_fitness = float(fitness[best_idx])
        fitness_trace = [rabbit_fitness]

        for iteration in range(self.iterations):
            mean_position = hawks.mean(axis=0)
            e1 = 2.0 * (1.0 - (iteration / max(self.iterations, 1)))
            for i in range(self.population_size):
                e0 = 2.0 * self.rng.random() - 1.0
                escaping_energy = e1 * e0
                candidate = hawks[i].copy()

                if abs(escaping_energy) >= 1.0:
                    q = self.rng.random()
                    rand_idx = int(self.rng.integers(0, self.population_size))
                    x_rand = hawks[rand_idx]
                    if q >= 0.5:
                        r1 = self.rng.random(dim)
                        r2 = self.rng.random(dim)
                        candidate = x_rand - r1 * np.abs(x_rand - 2.0 * r2 * candidate)
                    else:
                        r3 = self.rng.random(dim)
                        r4 = self.rng.random(dim)
                        candidate = (rabbit - mean_position) - r3 * (self.lower_bounds + r4 * (self.upper_bounds - self.lower_bounds))
                else:
                    j = 2.0 * (1.0 - self.rng.random())
                    r = self.rng.random()
                    if r >= 0.5 and abs(escaping_energy) >= 0.5:
                        candidate = rabbit - escaping_energy * np.abs(j * rabbit - candidate)
                    elif r >= 0.5 and abs(escaping_energy) < 0.5:
                        candidate = rabbit - escaping_energy * np.abs(rabbit - candidate)
                    elif r < 0.5 and abs(escaping_energy) >= 0.5:
                        y = rabbit - escaping_energy * np.abs(j * rabbit - candidate)
                        z = y + self.rng.normal(0.0, 1.0, dim) * self._levy(dim)
                        candidate = self._select_better(candidate, y, z, objective)
                    else:
                        y = rabbit - escaping_energy * np.abs(j * rabbit - mean_position)
                        z = y + self.rng.normal(0.0, 1.0, dim) * self._levy(dim)
                        candidate = self._select_better(candidate, y, z, objective)

                hawks[i] = np.clip(candidate, self.lower_bounds, self.upper_bounds)
                fitness[i] = objective(hawks[i])
                if fitness[i] < rabbit_fitness:
                    rabbit_fitness = float(fitness[i])
                    rabbit = hawks[i].copy()

            fitness_trace.append(rabbit_fitness)

        return HHOResult(best_vector=rabbit, best_fitness=rabbit_fitness, fitness_trace=fitness_trace)

    def _select_better(
        self,
        current: np.ndarray,
        first: np.ndarray,
        second: np.ndarray,
        objective: Callable[[np.ndarray], float],
    ) -> np.ndarray:
        first = np.clip(first, self.lower_bounds, self.upper_bounds)
        second = np.clip(second, self.lower_bounds, self.upper_bounds)
        current = np.clip(current, self.lower_bounds, self.upper_bounds)

        current_fitness = objective(current)
        first_fitness = objective(first)
        if first_fitness < current_fitness:
            current = first
            current_fitness = first_fitness
        second_fitness = objective(second)
        if second_fitness < current_fitness:
            current = second
        return current

    def _levy(self, dim: int) -> np.ndarray:
        sigma = (
            math.gamma(1 + self.beta)
            * math.sin(math.pi * self.beta / 2.0)
            / (math.gamma((1 + self.beta) / 2.0) * self.beta * 2 ** ((self.beta - 1.0) / 2.0))
        ) ** (1.0 / self.beta)
        u = self.rng.normal(0.0, sigma, size=dim)
        v = self.rng.normal(0.0, 1.0, size=dim)
        return u / np.power(np.abs(v), 1.0 / self.beta)
