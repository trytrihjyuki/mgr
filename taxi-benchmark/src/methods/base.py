"""Base class for pricing methods with dual acceptance evaluation."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from time import time

from ..core import get_logger


class BasePricingMethod(ABC):
    """Abstract base class for all pricing methods."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pricing method.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = get_logger(f"methods.{self.get_method_name()}")
        
        # Extract common parameters
        self.alpha = config.get('alpha', 18.0)
        self.s_taxi = config.get('s_taxi', 25.0)
        
        # Sigmoid parameters - aligned with Hikima paper
        self.sigmoid_beta = config.get('sigmoid_beta', 1.3)
        # gamma = (0.3*sqrt(3)/pi) from experiment_Sigmoid.py line 49 - EXACT match
        import math
        self.sigmoid_gamma = config.get('sigmoid_gamma', (0.3 * math.sqrt(3) / math.pi))
    
    @abstractmethod
    def compute_prices(
        self,
        scenario_data: Dict[str, Any],
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute prices for all requesters optimized for specific acceptance function.
        
        Args:
            scenario_data: Dictionary with scenario data
            acceptance_function: 'PL' or 'Sigmoid' - which function to optimize for
            
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
        Execute the pricing method separately for each acceptance function.
        This ensures proper alignment with Hikima's experimental setup.
        
        Args:
            scenario_data: Dictionary with scenario data
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with results for both acceptance functions
        """
        # Results dictionary
        results = {
            'method': self.get_method_name(),
            'num_requesters': scenario_data.get('num_requesters', 0),
            'num_taxis': scenario_data.get('num_taxis', 0),
        }
        
        # Get valuations
        valuations = scenario_data.get('trip_amounts', np.array([]))
        
        # CRITICAL: Optimize and evaluate separately for each acceptance function
        for acceptance_function in ['PL', 'Sigmoid']:
            func_start = time()
            self.logger.info(f"[{self.get_method_name()}] Starting {acceptance_function} optimization (N={len(valuations)}, sims={num_simulations})")
            
            # Compute prices optimized for this specific acceptance function
            price_start = time()
            prices = self.compute_prices(scenario_data, acceptance_function)
            price_time = time() - price_start
            
            self.logger.debug(f"[{self.get_method_name()}] {acceptance_function} price computation: {price_time:.3f}s")
            
            # Record computation time
            computation_time = time() - func_start
            
            # Compute acceptance probabilities with the same function
            prob_start = time()
            acceptance_probs = self._compute_acceptance_probs_specific(
                prices, valuations, acceptance_function
            )
            prob_time = time() - prob_start
            
            self.logger.debug(f"[{self.get_method_name()}] {acceptance_function} probability computation: {prob_time:.3f}s")
            
            # Run Monte Carlo simulations using Hikima's approach
            sim_start = time()
            if num_simulations > 1:
                # Multiple simulations
                revenues = []
                matching_rates = []
                acceptance_rates = []
                
                for sim_idx in range(num_simulations):
                    # Simulate acceptance decisions exactly like Hikima
                    accepted = np.zeros(len(acceptance_probs))
                    for i in range(len(acceptance_probs)):
                        tmp = np.random.rand()
                        if tmp < acceptance_probs[i]:
                            accepted[i] = 1
                    
                    # Calculate objective value using Hikima's value_eval function
                    opt_value, matched_edges, rewards = self._compute_objective_value_hikima(
                        prices, accepted, scenario_data['edge_weights'],
                        scenario_data['num_requesters'], scenario_data['num_taxis']
                    )
                    
                    # Calculate metrics like Hikima
                    revenue = opt_value  # This is the total objective value
                    matching_rate = len(matched_edges) / len(accepted) if len(accepted) > 0 else 0
                    acceptance_rate = np.mean(accepted)
                    
                    revenues.append(revenue)
                    matching_rates.append(matching_rate)
                    acceptance_rates.append(acceptance_rate)
                    
                    # Log progress for long simulations
                    if sim_idx > 0 and (sim_idx + 1) % 10 == 0:
                        elapsed = time() - sim_start
                        self.logger.debug(f"[{self.get_method_name()}] {acceptance_function} simulation {sim_idx+1}/{num_simulations} ({elapsed:.1f}s)")
                
                # Aggregate results
                func_results = {
                    'prices': prices.tolist() if len(prices) > 0 else [],
                    'acceptance_probs': acceptance_probs.tolist() if len(acceptance_probs) > 0 else [],
                    'avg_revenue': float(np.mean(revenues)),
                    'std_revenue': float(np.std(revenues)),
                    'avg_matching_rate': float(np.mean(matching_rates)),
                    'std_matching_rate': float(np.std(matching_rates)),
                    'avg_acceptance_rate': float(np.mean(acceptance_rates)),
                    'computation_time': computation_time,
                    'num_simulations': num_simulations
                }
            else:
                # Single run using Hikima's approach
                accepted = np.zeros(len(acceptance_probs))
                for i in range(len(acceptance_probs)):
                    tmp = np.random.rand()
                    if tmp < acceptance_probs[i]:
                        accepted[i] = 1
                
                # Calculate objective value using Hikima's value_eval function
                opt_value, matched_edges, rewards = self._compute_objective_value_hikima(
                    prices, accepted, scenario_data['edge_weights'],
                    scenario_data['num_requesters'], scenario_data['num_taxis']
                )
                
                func_results = {
                    'prices': prices.tolist() if len(prices) > 0 else [],
                    'acceptance_probs': acceptance_probs.tolist() if len(acceptance_probs) > 0 else [],
                    'avg_revenue': float(opt_value),
                    'std_revenue': 0.0,
                    'avg_matching_rate': float(len(matched_edges) / len(accepted)) if len(accepted) > 0 else 0,
                    'std_matching_rate': 0.0,
                    'avg_acceptance_rate': float(np.mean(accepted)) if len(accepted) > 0 else 0,
                    'computation_time': computation_time,
                    'num_simulations': 1
                }
            
            sim_time = time() - sim_start
            func_total_time = time() - func_start
            self.logger.info(f"[{self.get_method_name()}] {acceptance_function} completed in {func_total_time:.2f}s (sims: {sim_time:.2f}s, avg_revenue: ${func_results['avg_revenue']:.2f})")
            
            # Store results for this acceptance function
            results[acceptance_function] = func_results
        
        # Store common prices info (from PL optimization for backward compatibility)
        results['prices'] = results['PL']['prices']
        results['computation_time'] = results['PL']['computation_time'] + results['Sigmoid']['computation_time']
        
        return results
    
    def _compute_acceptance_probs(
        self,
        prices: np.ndarray,
        valuations: np.ndarray
    ) -> np.ndarray:
        """
        Compute acceptance probabilities (deprecated - use _compute_acceptance_probs_specific).
        """
        # Default to PL for backward compatibility
        return self._compute_acceptance_probs_specific(prices, valuations, 'PL')
    
    def _compute_acceptance_probs_specific(
        self,
        prices: np.ndarray,
        valuations: np.ndarray,
        acceptance_function: str
    ) -> np.ndarray:
        """
        Compute acceptance probabilities for specific acceptance function.
        
        Args:
            prices: Offered prices
            valuations: Customer valuations (trip amounts)
            acceptance_function: 'PL' or 'Sigmoid'
            
        Returns:
            Array of acceptance probabilities
        """
        if len(prices) == 0 or len(valuations) == 0:
            return np.array([])
        
        if acceptance_function == 'PL':
            # Piecewise linear acceptance function
            # P(accept) = -2/valuation * price + 3
            probs = -2.0 / valuations * prices + 3.0
            probs = np.clip(probs, 0, 1)
            
        elif acceptance_function == 'Sigmoid':
            # Sigmoid acceptance function - exact match to Hikima implementation
            # From experiment_Sigmoid.py line 739 and 782-783:
            # P(accept) = 1 - (1 / (1 + exp((-price + beta*valuation) / (gamma*valuation))))
            beta = self.sigmoid_beta
            gamma = self.sigmoid_gamma
            
            exponent = (-prices + beta * valuations) / (gamma * np.abs(valuations))
            probs = 1.0 - (1.0 / (1.0 + np.exp(exponent)))
            
        else:
            raise ValueError(f"Unknown acceptance function: {acceptance_function}")
        
        return probs
    
    def _simulate_matching(
        self,
        n_requesters: int,
        n_taxis: int,
        edge_weights: np.ndarray,
        acceptance_decisions: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Simulate matching between requesters and taxis.
        
        Args:
            n_requesters: Number of requesters
            n_taxis: Number of taxis
            edge_weights: Edge weight matrix
            acceptance_decisions: Binary acceptance decisions
            
        Returns:
            Tuple of (matched_requesters, matched_taxis)
        """
        # Simple greedy matching
        matched_requesters = []
        matched_taxis = []
        available_taxis = set(range(n_taxis))
        
        # Sort requesters by acceptance decision and edge weight
        requester_order = []
        for i in range(n_requesters):
            if acceptance_decisions[i]:
                best_taxi = max(available_taxis, key=lambda j: edge_weights[i, j]) if available_taxis else None
                if best_taxi is not None:
                    requester_order.append((edge_weights[i, best_taxi], i, best_taxi))
        
        requester_order.sort(reverse=True)
        
        for _, requester, taxi in requester_order:
            if taxi in available_taxis:
                matched_requesters.append(requester)
                matched_taxis.append(taxi)
                available_taxis.remove(taxi)
        
        return matched_requesters, matched_taxis
    
    def _calculate_metrics(
        self,
        prices: np.ndarray,
        acceptance_decisions: np.ndarray,
        matched_requesters: List[int]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            prices: Offered prices
            acceptance_decisions: Binary acceptance decisions
            matched_requesters: List of matched requester indices
            
        Returns:
            Dictionary with metrics
        """
        n_requesters = len(prices)
        
        # Revenue: sum of prices for matched requesters who accepted
        revenue = sum(prices[r] for r in matched_requesters if acceptance_decisions[r])
        
        # Matching rate: fraction of requesters matched
        matching_rate = len(matched_requesters) / n_requesters if n_requesters > 0 else 0
        
        # Acceptance rate: fraction of requesters who accepted
        acceptance_rate = np.mean(acceptance_decisions) if len(acceptance_decisions) > 0 else 0
        
        # Profit (simplified - could include costs)
        profit = revenue
        
        return {
            'revenue': float(revenue),
            'profit': float(profit),
            'matching_rate': float(matching_rate),
            'acceptance_rate': float(acceptance_rate),
            'num_matched': len(matched_requesters)
        } 

    def _compute_objective_value_hikima(
        self,
        prices: np.ndarray,
        acceptance_decisions: np.ndarray,
        edge_weights: np.ndarray,
        n_requesters: int,
        n_taxis: int
    ) -> Tuple[float, List, np.ndarray]:
        """
        Compute objective value using Hikima's exact value_eval function.
        
        This replicates the value_eval function from experiment_PL.py and experiment_Sigmoid.py
        (lines 172-195) to ensure 1:1 alignment.
        """
        import networkx as nx
        
        # Build bipartite graph exactly like Hikima
        group1 = range(n_requesters)
        group2 = range(n_requesters, n_requesters + n_taxis)
        g_post = nx.Graph()
        g_post.add_nodes_from(group1, bipartite=1)
        g_post.add_nodes_from(group2, bipartite=0)
        
        # Add edges for accepted requesters only
        for i in range(n_requesters):
            if acceptance_decisions[i] == 1:
                for j in range(n_taxis):
                    val = prices[i] + edge_weights[i, j]  # Hikima's line 182
                    g_post.add_edge(i, j + n_requesters, weight=val)
        
        # Find maximum weight matching (Hikima's line 183)
        matched_edges = nx.max_weight_matching(g_post)
        
        # Calculate objective value and rewards exactly like Hikima (lines 184-194)
        opt_value = 0.0
        reward = np.zeros(n_requesters)
        
        for (i, j) in matched_edges:
            # Handle edge order (Hikima's lines 186-192)
            if i > j:
                jtmp = j
                j = i - n_requesters
                i = jtmp
            else:
                j = j - n_requesters
            
            # Calculate value: price + edge_weight (Hikima's line 193)
            opt_value += prices[i] + edge_weights[i, j]
            reward[i] = prices[i] + edge_weights[i, j]
        
        return opt_value, matched_edges, reward 