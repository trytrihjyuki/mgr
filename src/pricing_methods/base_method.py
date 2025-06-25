"""
Base class for all pricing methods in the ride-hailing benchmark.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass


@dataclass
class PricingResult:
    """Result from a pricing method calculation."""
    method_name: str
    prices: np.ndarray
    acceptance_probabilities: np.ndarray
    objective_value: float
    computation_time: float
    matches: List[Tuple[int, int]]  # List of (requester_idx, taxi_idx) pairs
    additional_metrics: Dict[str, Any] = None


class BasePricingMethod(ABC):
    """
    Base class for all pricing methods.
    
    All pricing methods must implement the calculate_prices method
    and follow the same interface for fair benchmarking.
    """
    
    def __init__(self, method_name: str, **kwargs):
        self.method_name = method_name
        self.config = kwargs
        
    @abstractmethod
    def calculate_prices(self, 
                        requesters_data: pd.DataFrame,
                        taxis_data: pd.DataFrame,
                        distance_matrix: np.ndarray,
                        **kwargs) -> PricingResult:
        """
        Calculate optimal prices for the given scenario.
        
        Args:
            requesters_data: DataFrame with requester information
            taxis_data: DataFrame with taxi information  
            distance_matrix: Matrix of distances between requesters and taxis
            **kwargs: Method-specific parameters
            
        Returns:
            PricingResult with prices and metrics
        """
        pass
    
    def _calculate_edge_weights(self, 
                               distance_matrix: np.ndarray,
                               trip_distances: np.ndarray,
                               alpha: float = 18.0,
                               s_taxi: float = 25.0) -> np.ndarray:
        """
        Calculate edge weights W[i,j] as per Hikima methodology.
        
        W[i,j] = -(distance_ij + trip_distance_i) / s_taxi * alpha
        """
        n, m = distance_matrix.shape
        w_matrix = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                w_matrix[i, j] = -(distance_matrix[i, j] + trip_distances[i]) / s_taxi * alpha
                
        return w_matrix
    
    def _calculate_acceptance_probability(self,
                                        prices: np.ndarray,
                                        trip_amounts: np.ndarray,
                                        acceptance_function: str = 'PL') -> np.ndarray:
        """
        Calculate acceptance probabilities using Hikima formulas.
        
        Args:
            prices: Proposed prices
            trip_amounts: Historical trip amounts (reservation prices)
            acceptance_function: 'PL' for piecewise linear, 'Sigmoid' for sigmoid
            
        Returns:
            Array of acceptance probabilities
        """
        if acceptance_function == 'PL':
            # Piecewise Linear: from experiment_PL.py
            # p(price) = -2.0/trip_amount * price + 3
            c = 2.0 / trip_amounts
            d = 3.0
            acceptance_probs = -c * prices + d
            
        elif acceptance_function == 'Sigmoid':
            # Sigmoid: from experiment_sigmoid.py
            # p(price) = 1 - 1/(1 + exp((-price + beta*trip_amount)/(gamma*trip_amount)))
            beta = 1.3
            gamma = 0.3 * np.sqrt(3) / np.pi
            
            exponent = (-prices + beta * trip_amounts) / (gamma * trip_amounts)
            # Clip to avoid overflow
            exponent = np.clip(exponent, -50, 50)
            acceptance_probs = 1 - 1 / (1 + np.exp(exponent))
        else:
            raise ValueError(f"Unknown acceptance function: {acceptance_function}")
            
        # Ensure probabilities are in [0, 1]
        return np.clip(acceptance_probs, 0.0, 1.0)
    
    def _evaluate_matching(self,
                          prices: np.ndarray,
                          acceptance_results: np.ndarray,
                          w_matrix: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Evaluate the objective value for a given pricing and acceptance results.
        Uses maximum weight bipartite matching as per Hikima methodology.
        
        Args:
            prices: Proposed prices
            acceptance_results: Binary array indicating acceptance (1) or rejection (0)
            w_matrix: Edge weight matrix W[i,j]
            
        Returns:
            Tuple of (objective_value, matches)
        """
        import networkx as nx
        
        n, m = w_matrix.shape
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Add nodes: group1 = requesters (0 to n-1), group2 = taxis (n to n+m-1)
        G.add_nodes_from(range(n), bipartite=0)
        G.add_nodes_from(range(n, n+m), bipartite=1)
        
        # Add edges only for accepted requests
        for i in range(n):
            if acceptance_results[i] == 1:
                for j in range(m):
                    weight = prices[i] + w_matrix[i, j]
                    G.add_edge(i, n+j, weight=weight)
        
        # Find maximum weight matching
        matched_edges = nx.max_weight_matching(G)
        
        # Calculate objective value and extract matches
        objective_value = 0.0
        matches = []
        
        for edge in matched_edges:
            i, j_plus_n = edge
            if i > j_plus_n:  # Ensure consistent ordering
                i, j_plus_n = j_plus_n, i
            j = j_plus_n - n
            
            if 0 <= i < n and 0 <= j < m:
                objective_value += prices[i] + w_matrix[i, j]
                matches.append((i, j))
        
        return objective_value, matches 