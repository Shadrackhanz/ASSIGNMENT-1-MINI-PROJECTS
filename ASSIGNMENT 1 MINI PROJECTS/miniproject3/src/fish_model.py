"""Reusable classes and functions for Mini-Project 3."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import mean, median, variance, stdev
from typing import Iterable
import numpy as np

@dataclass
class FishStock:
    """Discrete logistic fish stock model with proportional harvesting."""
    initial_stock: float = 4000.0
    growth_rate: float = 0.4
    carrying_capacity: float = 10000.0
    harvest_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.initial_stock < 0 or self.carrying_capacity <= 0:
            raise ValueError("Stock must be non-negative and capacity positive")
        if self.initial_stock > self.carrying_capacity:
            raise ValueError("initial_stock cannot exceed carrying_capacity")
        if self.growth_rate < 0 or not 0 <= self.harvest_rate <= 1:
            raise ValueError("Invalid growth_rate or harvest_rate")

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Return stock values for weeks 0 through `weeks`, in tonnes."""
        if weeks < 0:
            raise ValueError("weeks must be non-negative")
        stock = np.zeros(weeks + 1, dtype=float)
        stock[0] = self.initial_stock
        for t in range(weeks):
            n = stock[t]
            stock[t + 1] = np.clip(
                n + self.growth_rate * n * (1 - n / self.carrying_capacity)
                - self.harvest_rate * n,
                0, self.carrying_capacity
            )
        return stock

    def weekly_harvest(self, stock: np.ndarray) -> np.ndarray:
        """Return weekly harvest in tonnes, based on stock at each week's start."""
        stock = np.asarray(stock, dtype=float)
        if stock.ndim != 1 or stock.size < 1 or np.any(stock < 0):
            raise ValueError("stock must be a non-empty, non-negative 1D array")
        return self.harvest_rate * stock[:-1]

@dataclass
class PriceModel:
    """Bounded seeded random walk for weekly price in UGX/kg."""
    start_price: float = 12000.0
    lower_bound: float = 9000.0
    upper_bound: float = 16000.0
    weekly_shock_sd: float = 500.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.lower_bound <= 0 or self.upper_bound <= self.lower_bound:
            raise ValueError("Invalid price bounds")
        if not self.lower_bound <= self.start_price <= self.upper_bound:
            raise ValueError("start_price must be within bounds")
        if self.weekly_shock_sd < 0:
            raise ValueError("weekly_shock_sd must be non-negative")

    def simulate(self, weeks: int = 52, seed: int | None = None) -> np.ndarray:
        """Return weekly prices, including the starting price."""
        if weeks < 1:
            raise ValueError("weeks must be at least 1")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        prices = np.empty(weeks, dtype=float)
        prices[0] = self.start_price
        for i, shock in enumerate(rng.normal(0, self.weekly_shock_sd, weeks - 1), 1):
            prices[i] = np.clip(prices[i - 1] + shock, self.lower_bound, self.upper_bound)
        return prices

def revenue_from_harvest(harvest_tonnes: np.ndarray, prices_ugx_per_kg: np.ndarray) -> np.ndarray:
    """Return weekly revenue in UGX."""
    harvest_tonnes = np.asarray(harvest_tonnes, dtype=float)
    prices_ugx_per_kg = np.asarray(prices_ugx_per_kg, dtype=float)
    if harvest_tonnes.ndim != 1 or prices_ugx_per_kg.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if len(harvest_tonnes) != len(prices_ugx_per_kg):
        raise ValueError("Inputs must have equal lengths")
    if np.any(harvest_tonnes < 0) or np.any(prices_ugx_per_kg < 0):
        raise ValueError("Inputs must be non-negative")
    return harvest_tonnes * 1000 * prices_ugx_per_kg

def revenue_summary(revenue: Iterable[float]) -> dict[str, float]:
    """Return mean, median, sample variance, standard deviation and CV percent."""
    values = [float(x) for x in revenue]
    if not values or any(x < 0 for x in values):
        raise ValueError("Revenue must be non-empty and non-negative")
    avg = mean(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean_UGX": avg, "median_UGX": median(values),
        "variance_UGX_squared": variance(values) if len(values) > 1 else 0.0,
        "std_dev_UGX": sd, "CV_percent": sd / avg * 100 if avg else 0.0
    }

@dataclass
class RiskAssessor:
    """Classify revenue risk using CV and calculate a Monte Carlo lower-tail VaR."""
    low_cv_percent: float = 10.0
    moderate_cv_percent: float = 20.0

    def __post_init__(self) -> None:
        if self.low_cv_percent < 0 or self.moderate_cv_percent <= self.low_cv_percent:
            raise ValueError("CV thresholds must be ordered and non-negative")

    def classify(self, revenue_values: Iterable[float]) -> str:
        """Return Low, Moderate or High based on coefficient of variation."""
        cv = revenue_summary(revenue_values)["CV_percent"]
        if cv < self.low_cv_percent:
            return "Low"
        if cv <= self.moderate_cv_percent:
            return "Moderate"
        return "High"

    def monte_carlo_var(self, harvest_tonnes: np.ndarray, price_model: PriceModel,
                        n_paths: int = 1000, seed: int = 2026) -> tuple[float, np.ndarray]:
        """Return the 5th percentile of simulated annual revenue and all path totals."""
        if n_paths < 1000:
            raise ValueError("n_paths must be at least 1000")
        rng = np.random.default_rng(seed)
        annual = np.empty(n_paths, dtype=float)
        for i in range(n_paths):
            path_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            prices = price_model.simulate(len(harvest_tonnes), seed=path_seed)
            annual[i] = revenue_from_harvest(harvest_tonnes, prices).sum()
        return float(np.quantile(annual, 0.05)), annual
