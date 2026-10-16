"""Reusable classes for Mini-Project 4: rainfall and crop suitability."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MONTHS = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

@dataclass
class Region:
    """Store a region's 12 monthly rainfall totals and calculate summary measures."""
    name: str
    rainfall_mm: np.ndarray

    def __post_init__(self) -> None:
        self.rainfall_mm = np.asarray(self.rainfall_mm, dtype=float)
        if self.rainfall_mm.ndim != 1 or len(self.rainfall_mm) != 12:
            raise ValueError("rainfall_mm must be a 1D array of 12 values")
        if not np.all(np.isfinite(self.rainfall_mm)) or np.any(self.rainfall_mm < 0):
            raise ValueError("rainfall values must be finite and non-negative")

    def annual_total(self) -> float:
        """Return annual rainfall total in mm."""
        return float(self.rainfall_mm.sum())

    def monthly_mean(self) -> float:
        """Return mean monthly rainfall in mm."""
        return float(self.rainfall_mm.mean())

    def wettest_month(self) -> tuple[str, float]:
        """Return wettest month and rainfall."""
        i = int(np.argmax(self.rainfall_mm))
        return str(MONTHS[i]), float(self.rainfall_mm[i])

    def driest_month(self) -> tuple[str, float]:
        """Return driest month and rainfall."""
        i = int(np.argmin(self.rainfall_mm))
        return str(MONTHS[i]), float(self.rainfall_mm[i])

    def coefficient_of_variation(self) -> float:
        """Return population coefficient of variation as a percentage."""
        avg = self.monthly_mean()
        return float(self.rainfall_mm.std(ddof=0) / avg * 100) if avg else 0.0

    def __repr__(self) -> str:
        return f"Region(name={self.name!r}, annual_total_mm={self.annual_total():.1f})"

@dataclass(frozen=True)
class CropRule:
    """Provisional crop rainfall screening range in mm/month."""
    crop_name: str
    min_monthly_mm: float
    max_monthly_mm: float
    source_note: str = "Illustrative threshold; replace with cited agronomy source."

    def __post_init__(self) -> None:
        if self.min_monthly_mm < 0 or self.max_monthly_mm <= self.min_monthly_mm:
            raise ValueError("Rainfall range must be non-negative and increasing")

    def classify(self, rainfall_mm: float) -> str:
        """Classify a month's rainfall as drought, suitable, or excess rainfall risk."""
        if not np.isfinite(rainfall_mm) or rainfall_mm < 0:
            raise ValueError("rainfall must be finite and non-negative")
        if rainfall_mm < self.min_monthly_mm:
            return "Drought risk"
        if rainfall_mm > self.max_monthly_mm:
            return "Excess rainfall risk"
        return "Suitable (rainfall screen)"

def cosine_similarity(x, y) -> float:
    """Compute cosine similarity for equal-length, non-zero 1D vectors."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Inputs must be 1D vectors of equal length")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Vectors must be finite")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")
    return float(np.dot(a, b) / denom)
