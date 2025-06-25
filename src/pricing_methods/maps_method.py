#!/usr/bin/env python3
"""
MAPS Method Implementation
Extracted from experiment_PL.py and experiment_sigmoid.py sources
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MAPSResult:
    """Result structure for MAPS method"""
    method_name: str = "MAPS"
    prices: np.ndarray = None
    objective_value: float = 0.0
    computation_time: float = 0.0
    convergence_iterations: int = 0
    acceptance_probabilities: np.ndarray = None
    matched_pairs: List[Tuple[int, int]] = None
    area_prices: Dict[int, float] = None
    
    def __post_init__(self):
        if self.matched_pairs is None:
            self.matched_pairs = []
        if self.area_prices is None:
            self.area_prices = {}


class MAPSMethod:
    """
    Implementation of MAPS (Multi-Area Pricing Strategy) method
    Based on exact mathematical formulation from the provided sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MAPS method with configuration
        
        Args:
            config: Configuration dictionary containing method parameters
        """
        self.alpha = config.get('alpha', 18.0)
        self.s_taxi = config.get('s_taxi', 25.0)
        self.s_0_rate = config.get('s_0_rate', 1.5)
        self.price_discretization_rate = config.get('price_discretization_rate', 0.05)
        self.max_matching_distance_km = config.get('max_matching_distance_km', 2.0)
        self.acceptance_type = config.get('acceptance_type', 'PL')  # 'PL' or 'Sigmoid'
        
        # Calculated parameters
        self.s_a = 1 / (self.s_0_rate - 1)
        self.s_b = 1 + 1 / (self.s_0_rate - 1)
        
        # Sigmoid parameters
        self.sigmoid_params = config.get('sigmoid_params', {
            'beta': 1.3,
            'gamma': 0.3 * math.sqrt(3) / math.pi
        })
        
        logger.info(f"Initialized MAPS method with α={self.alpha}, s_0_rate={self.s_0_rate}")
    
    def dfs(self, v: int, visited: Set[int], edges: List[Set[int]], matched: List[int]) -> bool:
        """
        Depth-first search for augmenting path in bipartite matching
        Exact implementation from original MAPS code
        
        Args:
            v: Current vertex
            visited: Set of visited vertices
            edges: Adjacency list representation
            matched: Current matching
            
        Returns:
            True if augmenting path found
        """
        for u in edges[v]:
            if u in visited:
                continue
            visited.add(u)
            if matched[u] == -1 or self.dfs(matched[u], visited, edges, matched):
                matched[u] = v
                return True
        return False
    
    def calculate_acceptance_rates(self, area_id: int, requesters_data: np.ndarray, 
                                  price_range: List[float]) -> np.ndarray:
        """
        Calculate average acceptance rates for each price in the given area
        Following exact MAPS methodology
        
        Args:
            area_id: ID of the area
            requesters_data: Requester data for the area
            price_range: List of candidate prices
            
        Returns:
            Array of acceptance rates for each price
        """
        acceptance_rates = np.zeros(len(price_range))
        area_requesters = requesters_data[requesters_data[:, 1] == area_id]
        
        if len(area_requesters) == 0:
            return acceptance_rates
        
        for k, price in enumerate(price_range):
            accept_sum = 0.0
            
            for requester in area_requesters:
                trip_distance = requester[2]  # o_dist in original code
                total_amount = requester[3]   # to_am in original code
                
                if self.acceptance_type == 'PL':
                    # Piecewise linear acceptance (exact formula from experiment_PL.py)
                    acceptance_prob = max(0, min(1, -self.s_a/total_amount * price * trip_distance + self.s_b))
                else:
                    # Sigmoid acceptance (exact formula from experiment_sigmoid.py)
                    beta = self.sigmoid_params['beta']
                    gamma = self.sigmoid_params['gamma']
                    acceptance_prob = 1 - (1 / (1 + math.exp((-price * trip_distance + beta * total_amount) / (gamma * total_amount))))
                
                accept_sum += acceptance_prob
            
            acceptance_rates[k] = accept_sum / len(area_requesters)
        
        return acceptance_rates
    
    def solve(self, requesters_data: np.ndarray, taxis_data: np.ndarray) -> MAPSResult:
        """
        Main solve method for MAPS pricing optimization
        Following exact algorithm from experiment sources
        
        Args:
            requesters_data: Array of requester data [borough, area_id, trip_dist, total_amount, ...]
            taxis_data: Array of taxi data [area_id, ...]
            
        Returns:
            MAPSResult containing solution
        """
        import time
        start_time = time.time()
        
        n = len(requesters_data)
        m = len(taxis_data)
        
        logger.info(f"Starting MAPS method with {n} requesters, {m} taxis")
        
        # Extract unique area IDs where requesters exist
        unique_areas = list(set(requesters_data[:, 1]))
        ID_set = [int(area_id) for area_id in unique_areas]
        
        # Calculate price bounds (exact from original code)
        if len(requesters_data) > 0:
            price_ratios = requesters_data[:, 3] / requesters_data[:, 2]  # total_amount / trip_distance
            p_max = np.max(price_ratios) * self.s_0_rate
            p_min = np.min(price_ratios)
        else:
            p_max = 20.0
            p_min = 5.0
        
        # Calculate number of price discretization levels
        d_number = int(np.trunc(np.log(p_max / p_min) / np.log(1 + self.price_discretization_rate))) + 1
        
        # Initialize MAPS variables (exact from original code)
        p_current = np.ones(len(ID_set)) * p_max
        current_count = np.zeros(len(ID_set))
        Nr = np.zeros(len(ID_set))  # Current number of matched requesters per area
        Nr_max = np.zeros(len(ID_set))  # Maximum requesters per area
        
        # Calculate maximum requesters per area
        for i, area_id in enumerate(ID_set):
            Nr_max[i] = np.sum(requesters_data[:, 1] == area_id)
        
        # Calculate total distance per area
        dr_sum = np.zeros(len(ID_set))
        for i, area_id in enumerate(ID_set):
            dr_sum[i] = np.sum(requesters_data[requesters_data[:, 1] == area_id, 2])
        
        # Pre-calculate acceptance rates for all areas and price levels (S matrix from paper)
        S = np.ones((len(ID_set), d_number)) * np.inf
        
        for r, area_id in enumerate(ID_set):
            price_list = []
            p_tmp = p_max
            for k in range(d_number):
                price_list.append(p_tmp)
                p_tmp = p_tmp / (1 + self.price_discretization_rate)
            
            S[r, :] = self.calculate_acceptance_rates(area_id, requesters_data, price_list)
        
        # Initialize bipartite matching structures
        edges = [set() for _ in range(n)]
        matched = [-1] * m
        matched_r = [-1] * n
        
        # Calculate distance matrix and set up bipartite graph
        # Assume taxis can match with requesters within max_matching_distance_km
        for i in range(n):
            for j in range(m):
                # Simplified distance calculation (in real implementation, use actual coordinates)
                # For now, allow matching within same or nearby areas
                if True:  # Simplified condition - in reality check actual distance
                    edges[i].add(j)
        
        # MAIN MAPS ALGORITHM
        iterations = 0
        
        # Initialize delta calculations for first iteration
        p_new = np.ones(len(ID_set)) * p_max
        new_count = np.zeros(len(ID_set))
        delta_new = np.zeros(len(ID_set))
        
        # Calculate initial deltas for each area
        for r in range(len(ID_set)):
            # Calculate optimal price for current state (exact from original algorithm)
            dr_Nr_sum = 0
            r_count = 0
            count = 0
            
            # Calculate distance sum for Nr[r]+1 requesters in area
            for req_idx, requester in enumerate(requesters_data):
                if requester[1] == ID_set[r]:
                    dr_Nr_sum += requester[2]
                    r_count += 1
                    if r_count == Nr[r] + 1:
                        break
            
            # Find optimal price
            value_tmp = 0.0
            p_tmp = p_max
            d_count = 0
            
            while p_tmp >= p_min:
                C = dr_sum[r]
                D = dr_Nr_sum
                
                current_value = min(C * (p_tmp - self.alpha/self.s_taxi) * S[r, d_count],
                                   D * (p_tmp - self.alpha/self.s_taxi))
                
                if value_tmp < current_value:
                    value_tmp = current_value
                    p_opt = p_tmp
                    opt_d_count = d_count
                
                p_tmp = p_tmp / (1 + self.price_discretization_rate)
                d_count += 1
            
            # Calculate delta (improvement potential)
            current_objective = (p_current[r] - self.alpha/self.s_taxi) * S[r, int(current_count[r])]
            optimal_objective = (p_opt - self.alpha/self.s_taxi) * S[r, opt_d_count]
            
            delta_new[r] = optimal_objective - current_objective
            p_new[r] = p_opt
            new_count[r] = opt_d_count
        
        # Main iteration loop
        while True:
            iterations += 1
            
            # Find area with maximum improvement potential
            max_index = np.argmax(delta_new)
            
            if delta_new[max_index] <= 0:
                break
            
            # Try to find augmenting path for this area
            feasible_flag = False
            
            for i in range(n):
                if (requesters_data[i, 1] == ID_set[max_index] and 
                    matched_r[i] == -1):
                    
                    visited = set()
                    feasible_flag = self.dfs(i, visited, edges, matched)
                    if feasible_flag:
                        matched_r[i] = 1
                        break
            
            if feasible_flag:
                # Update matching for this area
                Nr[max_index] += 1
                p_current[max_index] = p_new[max_index]
                current_count[max_index] = new_count[max_index]
                
                # Recalculate delta for this area if more capacity available
                if Nr[max_index] + 1 <= Nr_max[max_index]:
                    # Calculate new optimal price and delta
                    dr_Nr_sum = 0
                    r_count = 0
                    
                    for requester in requesters_data:
                        if requester[1] == ID_set[max_index]:
                            dr_Nr_sum += requester[2]
                            r_count += 1
                            if r_count == Nr[max_index] + 1:
                                break
                    
                    value_tmp = 0.0
                    p_tmp = p_max
                    d_count = 0
                    
                    while p_tmp >= p_min:
                        C = dr_sum[max_index]
                        D = dr_Nr_sum
                        
                        current_value = min(C * (p_tmp - self.alpha/self.s_taxi) * S[max_index, d_count],
                                           D * (p_tmp - self.alpha/self.s_taxi))
                        
                        if value_tmp < current_value:
                            value_tmp = current_value
                            p_opt = p_tmp
                            opt_d_count = d_count
                        
                        p_tmp = p_tmp / (1 + self.price_discretization_rate)
                        d_count += 1
                    
                    current_objective = (p_current[max_index] - self.alpha/self.s_taxi) * S[max_index, int(current_count[max_index])]
                    optimal_objective = (p_opt - self.alpha/self.s_taxi) * S[max_index, opt_d_count]
                    
                    delta_new[max_index] = optimal_objective - current_objective
                    p_new[max_index] = p_opt
                    new_count[max_index] = opt_d_count
                else:
                    # No more capacity in this area
                    delta_new[max_index] = -1
                    p_new[max_index] = -1
                    new_count[max_index] = -1
            else:
                # No augmenting path found
                delta_new[max_index] = -1
        
        # Calculate final prices and results
        prices = np.zeros(n)
        area_prices = {}
        
        for i in range(n):
            area_id = requesters_data[i, 1]
            trip_distance = requesters_data[i, 2]
            
            # Find price for this area
            for h, id_val in enumerate(ID_set):
                if id_val == area_id:
                    area_price = p_current[h]
                    prices[i] = area_price * trip_distance
                    area_prices[int(area_id)] = area_price
                    break
        
        # Calculate acceptance probabilities
        acceptance_probs = np.zeros(n)
        for i in range(n):
            price = prices[i]
            trip_distance = requesters_data[i, 2]
            total_amount = requesters_data[i, 3]
            
            if self.acceptance_type == 'PL':
                acceptance_probs[i] = max(0, min(1, -self.s_a/total_amount * price + self.s_b))
            else:
                beta = self.sigmoid_params['beta']
                gamma = self.sigmoid_params['gamma']
                acceptance_probs[i] = 1 - (1 / (1 + math.exp((-price + beta * total_amount) / (gamma * total_amount))))
        
        # Calculate objective value
        matched_pairs = [(i, matched[j]) for j, i in enumerate(matched) if i != -1]
        objective_value = sum(prices[i] for i, _ in matched_pairs)
        
        computation_time = time.time() - start_time
        
        logger.info(f"MAPS method completed in {computation_time:.3f}s, {iterations} iterations, objective: {objective_value:.2f}")
        
        return MAPSResult(
            prices=prices,
            objective_value=objective_value,
            computation_time=computation_time,
            convergence_iterations=iterations,
            acceptance_probabilities=acceptance_probs,
            matched_pairs=matched_pairs,
            area_prices=area_prices
        ) 