"""
Rideshare Pricing Optimization Benchmark Suite

A comprehensive benchmarking platform for comparing different pricing algorithms
in ride-hailing scenarios using real NYC taxi data.

Methods benchmarked:
1. MinMaxCost Flow (Hikima et al.)
2. MAPS Algorithm 
3. LinUCB Contextual Bandit
4. Linear Program (Gupta-Nagarajan)
"""

__version__ = "1.0.0"
__author__ = "Rideshare Pricing Research Team"

from .utils.config import Config
from .experiments.runner import ExperimentRunner
from .experiments.evaluator import ResultsEvaluator

__all__ = [
    "Config",
    "ExperimentRunner", 
    "ResultsEvaluator"
] 