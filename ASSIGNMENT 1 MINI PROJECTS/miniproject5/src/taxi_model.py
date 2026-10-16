"""Reusable code for the taxi route revenue and fleet planner."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean, variance, stdev
from typing import Sequence
import numpy as np
from scipy.linalg import solve

@dataclass
class Route:
    """Daily passenger journeys and fare for one route."""
    name: str
    passenger_counts: np.ndarray
    fare_ugx: float
    def __post_init__(self):
        self.passenger_counts=np.asarray(self.passenger_counts,dtype=float)
        if not self.name.strip(): raise ValueError("route name cannot be empty")
        if self.passenger_counts.ndim!=1 or len(self.passenger_counts)<2: raise ValueError("need at least two daily counts")
        if not np.all(np.isfinite(self.passenger_counts)) or np.any(self.passenger_counts<0): raise ValueError("counts must be finite and non-negative")
        if not np.isfinite(self.fare_ugx) or self.fare_ugx<0: raise ValueError("fare must be finite and non-negative")
    def daily_revenue(self, counts=None):
        values=self.passenger_counts if counts is None else np.asarray(counts,dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values<0): raise ValueError("counts must be finite and non-negative")
        return values*self.fare_ugx
    def total_revenue(self, counts=None): return float(np.sum(self.daily_revenue(counts)))
    def descriptive_statistics(self):
        x=self.passenger_counts.tolist()
        return {"mean":float(mean(x)),"variance":float(variance(x)),"standard_deviation":float(stdev(x))}

def _series(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.all(np.isfinite(x)) or np.any(x<0): raise ValueError("values must be a non-empty finite non-negative 1-D series")
    return x
def _horizon(h):
    if not isinstance(h,int) or h<1: raise ValueError("horizon must be a positive integer")

class Forecaster(ABC):
    """Abstract forecasting interface."""
    @abstractmethod
    def fit(self, values: Sequence[float]): ...
    @abstractmethod
    def predict(self, horizon: int=1) -> np.ndarray: ...

class MovingAverageForecaster(Forecaster):
    """Repeat the mean of the latest three observations."""
    def fit(self, values):
        self.values=_series(values)
        if len(self.values)<3: raise ValueError("moving average needs at least three observations")
        return self
    def predict(self,horizon=1):
        _horizon(horizon); return np.full(horizon,np.mean(self.values[-3:]))

class ExponentialSmoothingForecaster(Forecaster):
    """Simple exponential smoothing with alpha in (0, 1]."""
    def __init__(self,alpha=0.3):
        if not 0<alpha<=1: raise ValueError("alpha must be in (0, 1]")
        self.alpha=float(alpha)
    def fit(self,values):
        self.values=_series(values); level=float(self.values[0])
        for y in self.values[1:]: level=self.alpha*float(y)+(1-self.alpha)*level
        self.level=level; return self
    def predict(self,horizon=1):
        _horizon(horizon); return np.full(horizon,self.level)

class LinearTrendForecaster(Forecaster):
    """Least-squares linear trend using numpy.polyfit."""
    def fit(self,values):
        self.values=_series(values); x=np.arange(len(self.values),dtype=float)
        self.slope,self.intercept=np.polyfit(x,self.values,1); return self
    def predict(self,horizon=1):
        _horizon(horizon); x=np.arange(len(self.values),len(self.values)+horizon,dtype=float)
        return np.maximum(0,self.slope*x+self.intercept)

def solve_ntinda_equilibrium():
    """Solve Qd=120-.02P and Qs=10+.03P as a 2x2 linear system."""
    # Unknown vector is [P, Q].
    A=np.array([[0.02,1.0],[-0.03,1.0]])
    b=np.array([120.0,10.0])
    p,q=solve(A,b)
    return {"price_ugx":float(p),"quantity":float(q),"matrix":A,"rhs":b}

def forecast_factory(name,alpha=0.3):
    if name=="3-day moving average": return MovingAverageForecaster()
    if name=="simple exponential smoothing": return ExponentialSmoothingForecaster(alpha)
    if name=="linear trend": return LinearTrendForecaster()
    raise ValueError(f"unknown model: {name}")
