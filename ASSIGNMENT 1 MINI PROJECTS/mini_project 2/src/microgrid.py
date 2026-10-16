"""Reusable classes and functions for the Solar Micro-Grid Dispatch Planner."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, variance, stdev
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import nnls


@dataclass
class MicroGrid:
    """Solve daily solar/battery dispatch from two linear demand constraints.

    The system is:
        3x + 2y = D1
        4x +  y = D2
    where x is solar energy (kWh) and y is battery energy (kWh).
    """
    matrix: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.matrix is None:
            self.matrix = np.array([[3.0, 2.0], [4.0, 1.0]], dtype=float)
        else:
            self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.shape != (2, 2):
            raise ValueError("The coefficient matrix must be 2 by 2.")
        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("Matrix values must be finite.")

    @property
    def determinant(self) -> float:
        """Return the determinant; zero means no unique solution."""
        return float(np.linalg.det(self.matrix))

    @property
    def condition_number(self) -> float:
        """Return the 2-norm condition number; large values imply sensitivity."""
        return float(np.linalg.cond(self.matrix))

    def solve_day(self, d1: float, d2: float) -> np.ndarray:
        """Solve one day, rejecting missing, non-finite or negative demand."""
        values = np.asarray([d1, d2], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Demand values must be finite numbers.")
        if np.any(values < 0):
            raise ValueError("Demand values cannot be negative.")
        if abs(self.determinant) < 1e-12:
            raise ValueError("The system is singular and has no unique solution.")
        return solve(self.matrix, values, assume_a="gen")

    def solve_many(self, demands: np.ndarray) -> np.ndarray:
        """Solve a 2×N demand matrix in one vectorised linear-algebra call."""
        demands = np.asarray(demands, dtype=float)
        if demands.ndim != 2 or demands.shape[0] != 2:
            raise ValueError("demands must have shape (2, number_of_days).")
        if not np.all(np.isfinite(demands)) or np.any(demands < 0):
            raise ValueError("Demand values must be finite and non-negative.")
        if abs(self.determinant) < 1e-12:
            raise ValueError("The system is singular and has no unique solution.")
        return solve(self.matrix, demands, assume_a="gen")

    def solve_nonnegative(self, d1: float, d2: float) -> tuple[np.ndarray, float]:
        """Return a non-negative least-squares solution and residual norm."""
        values = np.asarray([d1, d2], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Demand values must be finite and non-negative.")
        result, residual = nnls(self.matrix, values)
        return result, float(residual)


def read_demands_csv(path: str) -> pd.DataFrame:
    """Read a demand CSV and validate the required columns and values."""
    frame = pd.read_csv(path)
    required = {"day", "d1_kwh", "d2_kwh"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    if frame.empty:
        raise ValueError("Demand CSV cannot be empty.")
    values = frame[["d1_kwh", "d2_kwh"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("CSV demands must be finite and non-negative.")
    return frame


def generate_demand_csv(path: str, seed: int = 2026, days: int = 30) -> pd.DataFrame:
    """Create reproducible illustrative demands with weekly pattern and noise."""
    if days < 1:
        raise ValueError("days must be at least 1.")
    rng = np.random.default_rng(seed)
    day_numbers = np.arange(1, days + 1)
    # Day 1 is treated as Monday; weekly pattern is illustrative, not measured.
    weekday = (day_numbers - 1) % 7
    weekly = np.array([0, 4, 7, 8, 5, -4, -8], dtype=float)
    d1 = 135 + weekly[weekday] + rng.normal(0, 5, days)
    d2 = 115 + 0.6 * weekly[weekday] + rng.normal(0, 4, days)
    frame = pd.DataFrame({
        "day": day_numbers,
        "weekday": np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[weekday],
        "d1_kwh": np.round(np.clip(d1, 1, None), 2),
        "d2_kwh": np.round(np.clip(d2, 1, None), 2),
    })
    frame.to_csv(path, index=False)
    return frame


def descriptive_stats(values: Iterable[float]) -> dict[str, float]:
    """Return sample mean, variance and standard deviation for non-empty data."""
    data = [float(v) for v in values]
    if not data:
        raise ValueError("At least one value is required.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Values must be finite.")
    return {
        "mean": mean(data),
        "variance": variance(data) if len(data) > 1 else 0.0,
        "std_dev": stdev(data) if len(data) > 1 else 0.0,
    }


def dispatch_report(demands: pd.DataFrame, grid: MicroGrid | None = None,
                    solar_rate: float = 150.0, battery_rate: float = 450.0) -> pd.DataFrame:
    """Solve each day and report raw feasibility, handled dispatch and cost.

    Negative unconstrained results are flagged. NNLS is used on flagged days;
    the non-negative result may not satisfy both equations exactly, so residual
    is retained as a transparent measure of unmet/inconsistent demand.
    """
    if solar_rate < 0 or battery_rate < 0:
        raise ValueError("Energy rates cannot be negative.")
    grid = grid or MicroGrid()
    required = {"day", "d1_kwh", "d2_kwh"}
    if not required.issubset(demands.columns):
        raise ValueError(f"Data must contain columns: {sorted(required)}")
    rows = []
    for row in demands.itertuples(index=False):
        raw = grid.solve_day(float(row.d1_kwh), float(row.d2_kwh))
        infeasible = bool(np.any(raw < -1e-9))
        if infeasible:
            used, residual = grid.solve_nonnegative(float(row.d1_kwh), float(row.d2_kwh))
            strategy = "NNLS non-negative approximation"
        else:
            used, residual = np.maximum(raw, 0.0), 0.0
            strategy = "Exact linear solution"
        solar, battery = map(float, used)
        cost = solar * solar_rate + battery * battery_rate
        rows.append({
            "day": int(row.day), "d1_kwh": float(row.d1_kwh), "d2_kwh": float(row.d2_kwh),
            "raw_solar_kwh": float(raw[0]), "raw_battery_kwh": float(raw[1]),
            "infeasible_raw_solution": infeasible, "solar_kwh": solar, "battery_kwh": battery,
            "handling": strategy, "residual_norm": residual, "daily_cost_ugx": cost,
        })
    return pd.DataFrame(rows)
