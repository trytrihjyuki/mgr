"""
MAPS (Area-based Pricing) Implementation

This implements the MAPS algorithm extracted from the provided source code
(experiment_PL.py and experiment_sigmoid.py). MAPS groups requesters by area
and optimizes pricing per area using bipartite matching.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Tuple, Set

from .base_method import BasePricingMethod, PricingResult


class MAPS(BasePricingMethod):
    """
    Implementation of the MAPS (Area-based Pricing) algorithm.
    
    This algorithm groups requesters by pickup area and optimizes prices
    per area using bipartite matching with augmented paths.
    """
    
    def __init__(self, **kwargs):
        super().__init__("MAPS", **kwargs)
        
        # MAPS parameters from the original source code
        self.alpha = kwargs.get('alpha', 18.0)
        self.s_taxi = kwargs.get('s_taxi', 25.0)
        self.s_0_rate = kwargs.get('s_0_rate', 1.5)
        self.price_discretization_rate = kwargs.get('price_discretization_rate', 0.05)
        self.max_matching_distance_km = kwargs.get('max_matching_distance_km', 2.0)
        
        # Acceptance function parameters
        self.acceptance_function = kwargs.get('acceptance_function', 'PL')
        
        # PL-specific parameters
        self.s_a = 1 / (self.s_0_rate - 1)  # From source code
        self.s_b = 1 + 1 / (self.s_0_rate - 1)
        
        # Sigmoid parameters
        self.sigmoid_beta = kwargs.get('sigmoid_beta', 1.3)
        self.sigmoid_gamma = kwargs.get('sigmoid_gamma', 0.3 * np.sqrt(3) / np.pi)

    def calculate_prices(self, 
                        requesters_data: pd.DataFrame,
                        taxis_data: pd.DataFrame,
                        distance_matrix: np.ndarray,
                        **kwargs) -> PricingResult:
        """
        Calculate optimal prices using MAPS algorithm.
        
        This implements the exact MAPS algorithm from the provided source code.
        """
        start_time = time.time()
        
        n = len(requesters_data)
        m = len(taxis_data)
        
        if n == 0 or m == 0:
            return self._create_empty_result(start_time)
        
        # Extract trip data
        trip_distances = requesters_data['trip_distance'].values * 1.60934  # Convert to km
        trip_amounts = requesters_data['total_amount'].values
        
        # Get area information (PULocationID or equivalent)
        if 'PULocationID' in requesters_data.columns:
            area_ids = requesters_data['PULocationID'].values
        elif 'pickup_zone_id' in requesters_data.columns:
            area_ids = requesters_data['pickup_zone_id'].values
        else:
            # Default to single area if no area information
            area_ids = np.ones(n, dtype=int)
        
        # Run MAPS algorithm
        prices = self._run_maps_algorithm(
            area_ids, trip_distances, trip_amounts, distance_matrix, n, m)
        
        # Calculate acceptance probabilities
        acceptance_probs = self._calculate_acceptance_probability(
            prices, trip_amounts, self.acceptance_function)
        
        # Calculate edge weights for evaluation
        w_matrix = self._calculate_edge_weights(distance_matrix, trip_distances, 
                                               self.alpha, self.s_taxi)
        
        # Simulate matching for evaluation
        acceptance_results = np.random.binomial(1, acceptance_probs)
        objective_value, matches = self._evaluate_matching(prices, acceptance_results, w_matrix)
        
        computation_time = time.time() - start_time
        
        return PricingResult(
            method_name=self.method_name,
            prices=prices,
            acceptance_probabilities=acceptance_probs,
            objective_value=objective_value,
            computation_time=computation_time,
            matches=matches,
            additional_metrics={
                'algorithm': 'area_based_bipartite_matching',
                'acceptance_function': self.acceptance_function,
                'n_areas': len(set(area_ids)),
                'n_requesters': n,
                'n_taxis': m
            }
        )
    
    def _run_maps_algorithm(self, 
                           area_ids: np.ndarray,
                           trip_distances: np.ndarray, 
                           trip_amounts: np.ndarray,
                           distance_matrix: np.ndarray,
                           n: int, m: int) -> np.ndarray:
        """
        Run the MAPS algorithm extracted from the provided source code.
        
        This follows the exact algorithm from experiment_PL.py and experiment_sigmoid.py.
        """
        # Get unique area IDs (ID_set in the source code)
        id_set = list(set(area_ids))
        num_areas = len(id_set)
        
        # Calculate price bounds
        p_max = np.max(trip_amounts / trip_distances) * self.s_0_rate
        p_min = np.min(trip_amounts / trip_distances)
        
        # Calculate number of price discretization levels
        d_number = int(np.log(p_max / p_min) / np.log(1 + self.price_discretization_rate)) + 1
        
        # Initialize MAPS variables
        p_current = np.ones(num_areas) * p_max
        current_count = np.zeros(num_areas)
        nr = np.zeros(num_areas)  # Number of matched requesters per area
        nr_max = np.zeros(num_areas)  # Maximum requesters per area
        
        # Calculate maximum requesters per area
        for i, area_id in enumerate(id_set):
            nr_max[i] = np.sum(area_ids == area_id)
        
        # Calculate distance sums per area (dr_sum in source code)
        dr_sum = np.zeros(num_areas)
        for i, area_id in enumerate(id_set):
            area_mask = (area_ids == area_id)
            dr_sum[i] = np.sum(trip_distances[area_mask])
        
        # Precompute acceptance probabilities for each area and price level (S matrix)
        S = self._precompute_acceptance_matrix(id_set, area_ids, trip_distances, 
                                              trip_amounts, p_max, d_number)
        
        # Initialize bipartite matching data structures
        edges = self._build_bipartite_edges(distance_matrix, n, m)
        matched = [-1] * m  # Which requester each taxi is matched to (-1 if unmatched)
        matched_r = [-1] * n  # Which taxi each requester is matched to (-1 if unmatched)
        
        # Initialize pricing optimization variables
        p_new = np.ones(num_areas) * p_max
        new_count = np.zeros(num_areas)
        delta_new = np.zeros(num_areas)
        
        # Calculate initial deltas for each area
        for r in range(num_areas):
            area_id = id_set[r]
            delta_new[r] = self._calculate_area_delta(
                r, area_id, area_ids, trip_distances, nr[r], dr_sum[r], 
                p_max, p_min, S, p_current[r], int(current_count[r]))
            p_new[r], new_count[r] = self._find_optimal_price_for_area(
                r, area_id, area_ids, trip_distances, nr[r], dr_sum[r], 
                p_max, p_min, S)
        
        # Run MAPS main algorithm loop
        iteration = 0
        max_iterations = min(n, 1000)  # Prevent infinite loops
        
        while iteration < max_iterations:
            # Find area with maximum delta improvement
            max_index = np.argmax(delta_new)
            
            if delta_new[max_index] <= 0:
                break  # No more improvements possible
                
            # Try to find an augmenting path for this area
            feasible_flag = False
            area_id = id_set[max_index]
            
            # Find an unmatched requester from the selected area
            for i in range(n):
                if area_ids[i] == area_id and matched_r[i] == -1:
                    # Try to find augmenting path using DFS
                    visited = set()
                    feasible_flag = self._dfs_augmenting_path(i, edges, matched, visited)
                    if feasible_flag:
                        matched_r[i] = 1  # Mark as matched
                        break
            
            if feasible_flag:
                # Update matching count and prices for this area
                nr[max_index] += 1
                p_current[max_index] = p_new[max_index]
                current_count[max_index] = new_count[max_index]
                
                # Recalculate delta for this area
                if nr[max_index] + 1 <= nr_max[max_index]:
                    delta_new[max_index] = self._calculate_area_delta(
                        max_index, area_id, area_ids, trip_distances, 
                        nr[max_index], dr_sum[max_index], p_max, p_min, S,
                        p_current[max_index], int(current_count[max_index]))
                    p_new[max_index], new_count[max_index] = self._find_optimal_price_for_area(
                        max_index, area_id, area_ids, trip_distances, 
                        nr[max_index], dr_sum[max_index], p_max, p_min, S)
                else:
                    delta_new[max_index] = -1  # No more capacity
                    p_new[max_index] = -1
                    new_count[max_index] = -1
            else:
                delta_new[max_index] = -1  # No augmenting path found
            
            iteration += 1
        
        # Convert area-based prices to requester-based prices
        prices = np.zeros(n)
        for i in range(n):
            area_id = area_ids[i]
            area_idx = id_set.index(area_id)
            prices[i] = p_current[area_idx] * trip_distances[i]
        
        return prices
    
    def _precompute_acceptance_matrix(self, id_set: List, area_ids: np.ndarray,
                                     trip_distances: np.ndarray, trip_amounts: np.ndarray,
                                     p_max: float, d_number: int) -> np.ndarray:
        """
        Precompute acceptance probabilities for each area and price level.
        
        This is the S matrix computation from the source code.
        """
        num_areas = len(id_set)
        S = np.ones((num_areas, d_number)) * np.inf
        
        for r, area_id in enumerate(id_set):
            area_mask = (area_ids == area_id)
            area_distances = trip_distances[area_mask]
            area_amounts = trip_amounts[area_mask]
            
            p_tmp = p_max
            for k in range(d_number):
                accept_sum = 0
                
                for dist, amount in zip(area_distances, area_amounts):
                    if self.acceptance_function == 'PL':
                        # PL acceptance: -s_a/amount * p_tmp * dist + s_b
                        acceptance_prob = -self.s_a / amount * p_tmp * dist + self.s_b
                        acceptance_prob = max(0, min(1, acceptance_prob))
                    else:
                        # Sigmoid acceptance
                        exponent = (-p_tmp * dist + self.sigmoid_beta * amount) / (self.sigmoid_gamma * amount)
                        exponent = np.clip(exponent, -50, 50)
                        acceptance_prob = 1 - 1 / (1 + np.exp(exponent))
                    
                    accept_sum += acceptance_prob
                
                S[r, k] = accept_sum / len(area_distances) if len(area_distances) > 0 else 0
                p_tmp = p_tmp / (1 + self.price_discretization_rate)
        
        return S
    
    def _build_bipartite_edges(self, distance_matrix: np.ndarray, n: int, m: int) -> List[Set[int]]:
        """
        Build bipartite graph edges based on distance constraints.
        
        From source code: "assume that it is possible to match a taxi with a requester within 2 km"
        """
        edges = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                if distance_matrix[i, j] <= self.max_matching_distance_km:
                    edges[i].add(j)
        
        return edges
    
    def _dfs_augmenting_path(self, v: int, edges: List[Set[int]], 
                            matched: List[int], visited: Set[int]) -> bool:
        """
        DFS to find augmenting paths in bipartite matching.
        
        This is the exact DFS implementation from the source code.
        """
        for u in edges[v]:
            if u in visited:
                continue
            visited.add(u)
            if matched[u] == -1 or self._dfs_augmenting_path(matched[u], edges, matched, visited):
                matched[u] = v
                return True
        return False
    
    def _calculate_area_delta(self, area_idx: int, area_id: int, area_ids: np.ndarray,
                             trip_distances: np.ndarray, nr_current: float, dr_sum: float,
                             p_max: float, p_min: float, S: np.ndarray,
                             p_current: float, current_count: int) -> float:
        """
        Calculate delta (improvement potential) for an area.
        
        This implements the delta calculation from the MAPS source code.
        """
        # Calculate D (distance sum for nr+1 requesters)
        area_mask = (area_ids == area_id)
        area_distances = trip_distances[area_mask]
        area_distances_sorted = np.sort(area_distances)
        
        if len(area_distances_sorted) <= nr_current + 1:
            D = np.sum(area_distances_sorted)
        else:
            D = np.sum(area_distances_sorted[:int(nr_current + 1)])
        
        # Find optimal price and count
        p_opt, opt_count = self._find_optimal_price_for_area(
            area_idx, area_id, area_ids, trip_distances, nr_current, dr_sum, p_max, p_min, S)
        
        # Calculate delta improvement
        if opt_count >= 0 and area_idx < len(S):
            new_value = (p_opt - self.alpha/self.s_taxi) * S[area_idx, int(opt_count)]
            current_value = (p_current - self.alpha/self.s_taxi) * S[area_idx, current_count] if current_count < len(S[area_idx]) else 0
            delta = new_value - current_value
        else:
            delta = -1
            
        return delta
    
    def _find_optimal_price_for_area(self, area_idx: int, area_id: int, area_ids: np.ndarray,
                                    trip_distances: np.ndarray, nr_current: float, dr_sum: float,
                                    p_max: float, p_min: float, S: np.ndarray) -> Tuple[float, int]:
        """
        Find the optimal price for an area given current matching state.
        
        This implements the price optimization logic from the MAPS source code.
        """
        # Calculate C and D values
        C = dr_sum
        
        # Calculate D (distance sum for nr+1 requesters)  
        area_mask = (area_ids == area_id)
        area_distances = trip_distances[area_mask]
        area_distances_sorted = np.sort(area_distances)
        
        if len(area_distances_sorted) <= nr_current + 1:
            D = np.sum(area_distances_sorted)
        else:
            D = np.sum(area_distances_sorted[:int(nr_current + 1)])
        
        # Search for optimal price
        value_tmp = 0.0
        p_tmp = p_max
        d_count = 0
        p_opt = p_max
        opt_d_count = 0
        
        while p_tmp >= p_min and d_count < S.shape[1]:
            if area_idx < len(S):
                # Calculate objective value for this price
                value_c = C * (p_tmp - self.alpha/self.s_taxi) * S[area_idx, d_count]
                value_d = D * (p_tmp - self.alpha/self.s_taxi)
                current_value = min(value_c, value_d)
                
                if current_value > value_tmp:
                    value_tmp = current_value
                    p_opt = p_tmp
                    opt_d_count = d_count
                    
            p_tmp = p_tmp / (1 + self.price_discretization_rate)
            d_count += 1
        
        return p_opt, opt_d_count
    
    def _create_empty_result(self, start_time):
        """Create empty result for edge cases."""
        return PricingResult(
            method_name=self.method_name,
            prices=np.array([]),
            acceptance_probabilities=np.array([]),
            objective_value=0.0,
            computation_time=time.time() - start_time,
            matches=[],
            additional_metrics={'empty_scenario': True}
        ) 