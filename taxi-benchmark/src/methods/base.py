"""Base class for pricing methods."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx
from time import time

from ..core.logging import get_logger
from ..core.types import ScenarioResult, PricingMethod


class BasePricingMethod(ABC):
    """Abstract base class for all pricing methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pricing method.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(f"method.{self.__class__.__name__}")
        
        # Common parameters
        self.alpha = config.get('alpha', 18.0)
        self.s_taxi = config.get('s_taxi', 25.0)
        self.epsilon = config.get('epsilon', 1e-10)
    
    @abstractmethod
    def compute_prices(
        self,
        scenario_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute prices for all requesters.
        
        Args:
            scenario_data: Dictionary with scenario data
            
        Returns:
            Array of prices for each requester
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of the pricing method."""
        pass
    
    def execute(
        self,
        scenario_data: Dict[str, Any],
        num_simulations: int = 1
    ) -> Dict[str, Any]:
        """
        Execute the pricing method and evaluate results for both PL and Sigmoid acceptance functions.
        
        Args:
            scenario_data: Dictionary with scenario data
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with results for both acceptance functions
        """
        # Start timing
        start_time = time()
        
        # Compute prices
        prices = self.compute_prices(scenario_data)
        
        # Record computation time
        computation_time = time() - start_time
        
        # Results dictionary for both acceptance functions
        dual_results = {}
        
        # Evaluate for both acceptance functions
        for acceptance_function in ['PL', 'Sigmoid']:
            # Compute acceptance probabilities
            acceptance_probs = self._compute_acceptance_probs_specific(
                prices,
                scenario_data['trip_amounts'],
                acceptance_function
            )
            
            # Run simulations
            results = []
            for i in range(num_simulations):
                # Simulate acceptance
                acceptances = self._simulate_acceptance(acceptance_probs)
                
                # Compute matching
                matching_result = self._compute_matching(
                    prices,
                    acceptances,
                    scenario_data
                )
                
                results.append(matching_result)
            
            # Aggregate results
            aggregated = self._aggregate_results(results, scenario_data)
            aggregated['acceptance_probs'] = acceptance_probs
            aggregated['acceptance_function'] = acceptance_function
            
            # Store results for this acceptance function
            dual_results[acceptance_function] = aggregated
        
        # Create combined results
        combined_results = {
            'computation_time': computation_time,
            'prices': prices,
            'method': self.get_method_name(),
            'PL': dual_results['PL'],
            'Sigmoid': dual_results['Sigmoid']
        }
        
        return combined_results
    
    def _compute_acceptance_probs(
        self,
        prices: np.ndarray,
        valuations: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceptance probabilities.
        
        Default implementation uses the acceptance function from config.
        This is kept for backward compatibility but new code should use
        _compute_acceptance_probs_specific.
        """
        acceptance_function = self.config.get('acceptance_function', 'PL')
        return self._compute_acceptance_probs_specific(prices, valuations, acceptance_function)
    
    def _compute_acceptance_probs_specific(
        self,
        prices: np.ndarray,
        valuations: np.ndarray,
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute acceptance probabilities for a specific acceptance function.
        
        Args:
            prices: Array of prices
            valuations: Array of valuations (trip amounts)
            acceptance_function: 'PL' or 'Sigmoid'
            
        Returns:
            Array of acceptance probabilities
        """
        if acceptance_function == 'PL':
            # Piecewise linear: P(accept) = -2/valuation * price + 3
            probs = -2.0 / valuations * prices + 3.0
            probs = np.clip(probs, 0, 1)
        else:  # Sigmoid
            beta = self.config.get('sigmoid_beta', 1.3)
            gamma = self.config.get('sigmoid_gamma', 0.276)
            exponent = (beta * valuations - prices) / (gamma * np.abs(valuations))
            probs = 1.0 / (1.0 + np.exp(-exponent))
        
        return probs
    
    def _simulate_acceptance(
        self,
        acceptance_probs: np.ndarray
    ) -> np.ndarray:
        """Simulate customer acceptance decisions."""
        random_vals = np.random.rand(len(acceptance_probs))
        return (random_vals < acceptance_probs).astype(int)
    
    def _compute_matching(
        self,
        prices: np.ndarray,
        acceptances: np.ndarray,
        scenario_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute optimal matching given prices and acceptances.
        
        Uses networkx for maximum weight matching.
        """
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        edge_weights = scenario_data['edge_weights']
        
        # Create bipartite graph
        G = nx.Graph()
        
        # Add nodes
        requesters = range(n_requesters)
        taxis = range(n_requesters, n_requesters + n_taxis)
        G.add_nodes_from(requesters, bipartite=0)
        G.add_nodes_from(taxis, bipartite=1)
        
        # Add edges for accepted requesters
        for i in range(n_requesters):
            if acceptances[i] == 1:
                for j in range(n_taxis):
                    # Weight is price minus cost (negative of edge_weight)
                    weight = prices[i] + edge_weights[i, j]
                    if weight > 0:  # Only add profitable edges
                        G.add_edge(i, n_requesters + j, weight=weight)
        
        # Find maximum weight matching
        if G.number_of_edges() > 0:
            matching = nx.max_weight_matching(G, maxcardinality=False)
            matched_edges = list(matching)
        else:
            matched_edges = []
        
        # Calculate metrics
        total_revenue = 0
        total_cost = 0
        num_matched = len(matched_edges)
        matched_requesters = set()
        
        for (i, j) in matched_edges:
            if i > j:
                i, j = j, i
            j = j - n_requesters
            
            total_revenue += prices[i]
            total_cost += -edge_weights[i, j]  # Cost is negative of weight
            matched_requesters.add(i)
        
        profit = total_revenue - total_cost
        
        return {
            'num_matched': num_matched,
            'total_revenue': total_revenue,
            'total_cost': total_cost,
            'profit': profit,
            'matched_edges': matched_edges,
            'acceptance_rate': np.mean(acceptances),
            'matching_rate': num_matched / n_requesters if n_requesters > 0 else 0
        }
    
    def _aggregate_results(
        self,
        results: list,
        scenario_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate results from multiple simulations."""
        if not results:
            return {
                'num_requesters': scenario_data['num_requesters'],
                'num_taxis': scenario_data['num_taxis'],
                'num_matched': 0,
                'total_revenue': 0,
                'total_cost': 0,
                'profit': 0,
                'acceptance_rate': 0,
                'matching_rate': 0
            }
        
        # Calculate averages
        aggregated = {
            'num_requesters': scenario_data['num_requesters'],
            'num_taxis': scenario_data['num_taxis'],
            'num_matched': np.mean([r['num_matched'] for r in results]),
            'total_revenue': np.mean([r['total_revenue'] for r in results]),
            'total_cost': np.mean([r['total_cost'] for r in results]),
            'profit': np.mean([r['profit'] for r in results]),
            'acceptance_rate': np.mean([r['acceptance_rate'] for r in results]),
            'matching_rate': np.mean([r['matching_rate'] for r in results]),
            
            # Also include standard deviations
            'profit_std': np.std([r['profit'] for r in results]),
            'matching_rate_std': np.std([r['matching_rate'] for r in results])
        }
        
        return aggregated 