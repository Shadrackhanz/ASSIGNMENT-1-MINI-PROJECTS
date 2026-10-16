from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
class DistrictPopulation:
    def __init__(self, name: str, years, population):
        self.name=name; self.years=np.asarray(years,dtype=int); self.population=np.asarray(population,dtype=float)
        if len(self.years)==0 or len(self.years)!=len(self.population): raise ValueError('years and population must be non-empty and equal length')
        if np.any(self.population<0) or len(np.unique(self.years))!=len(self.years): raise ValueError('population must be non-negative and years unique')
    def __len__(self): return len(self.years)
    def __repr__(self): return f'DistrictPopulation({self.name!r}, {len(self)} observations)'
    def growth_rates(self): return self.population[1:]/self.population[:-1]-1
    def cagr(self): return (self.population[-1]/self.population[0])**(1/(len(self)-1))-1
class Forecaster(ABC):
    def __init__(self): self.coef=None
    @abstractmethod
    def fit(self, years, values): ...
    @abstractmethod
    def predict(self, years): ...
class LinearTrend(Forecaster):
    def fit(self, years, values): self.coef=np.polyfit(np.asarray(years),np.asarray(values),1); return self
    def predict(self, years): return np.polyval(self.coef,np.asarray(years))
class CAGRForecaster(Forecaster):
    def fit(self, years, values):
        years=np.asarray(years); values=np.asarray(values,dtype=float)
        if np.any(values<=0): raise ValueError('CAGR model requires positive values')
        self.last_year=years[-1]; self.last_value=values[-1]; self.rate=(values[-1]/values[0])**(1/(len(values)-1))-1; return self
    def predict(self, years): return self.last_value*(1+self.rate)**(np.asarray(years)-self.last_year)
class FibonacciRatioForecaster(Forecaster):
    def fit(self, years, values):
        values=np.asarray(values,dtype=float)
        if len(values)<2 or np.any(values<=0): raise ValueError('requires at least two positive observations')
        self.last=values[-1]; self.ratio=values[-1]/values[-2]; self.year=np.asarray(years)[-1]; return self
    def predict(self, years):
        h=np.asarray(years)-self.year
        return self.last*np.power(self.ratio,np.maximum(h,0))
def metrics(actual,pred):
    actual=np.asarray(actual,dtype=float); pred=np.asarray(pred,dtype=float); e=actual-pred
    return {'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'MAPE_%':float(np.mean(np.abs(e/actual))*100)}
