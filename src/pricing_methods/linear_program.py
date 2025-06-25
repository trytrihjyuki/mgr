"""
Linear Program (Gupta-Nagarajan) Implementation

This implements the Linear Program approach using the provided PuLP implementation
of the Gupta-Nagarajan linear program for ride-hailing pricing optimization.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Tuple
import pulp as pl

from .base_method import BasePricingMethod, PricingResult


class LinearProgram(BasePricingMethod):
    """
    Implementation of the Gupta-Nagarajan Linear Program for ride-hailing pricing.
    
    This is a drop-in implementation of the Gupta-Nagarajan linear program,
    where every symbol corresponds exactly to the notation used in the theory.
    """
    
    def __init__(self, **kwargs):
        super().__init__("LinearProgram", **kwargs)
        
        # LP parameters
        self.min_price_factor = kwargs.get('min_price_factor', 0.5)
        self.max_price_factor = kwargs.get('max_price_factor', 2.0)
        self.price_grid_size = kwargs.get('price_grid_size', 10)
        self.solver_name = kwargs.get('solver_name', 'PULP_CBC_CMD')
        self.solver_timeout = kwargs.get('solver_timeout', 300)
        self.solver_verbose = kwargs.get('solver_verbose', False)
        
        # Acceptance function
        self.acceptance_function = kwargs.get('acceptance_function', 'sigmoid')
        
        # Sigmoid parameters
        self.sigmoid_beta = kwargs.get('sigmoid_beta', 1.3)
        self.sigmoid_gamma = kwargs.get('sigmoid_gamma', 0.3 * np.sqrt(3) / np.pi)
    
    def calculate_prices(self, 
                        requesters_data: pd.DataFrame,
                        taxis_data: pd.DataFrame,
                        distance_matrix: np.ndarray,
                        **kwargs) -> PricingResult:
        """
        Calculate optimal prices using Gupta-Nagarajan Linear Program.
        
        This implements the exact LP from the provided PuLP implementation.
        """
        start_time = time.time()
        
        n = len(requesters_data)
        m = len(taxis_data)
        
        if n == 0 or m == 0:
            return self._create_empty_result(start_time)
        
        # Extract trip data
        trip_distances = requesters_data['trip_distance'].values * 1.60934  # Convert to km
        trip_amounts = requesters_data['total_amount'].values
        
        # Define clients and taxis
        clients = list(range(n))
        taxis = list(range(m))
        
        # Define feasible edges (all pairs within reasonable distance)
        edges = set()
        for i in range(n):
            for j in range(m):
                if distance_matrix[i, j] <= 10.0:  # 10km max distance
                    edges.add((i, j))
        
        # Create price grids for each client
        price_grid = self._create_price_grids(clients, trip_amounts, trip_distances)
        
        # Calculate acceptance probabilities
        acceptance_prob = self._calculate_acceptance_probabilities(
            clients, price_grid, trip_amounts)
        
        # Calculate costs (driving costs)
        cost = self._calculate_costs(edges, distance_matrix, trip_distances)
        
        # Build and solve the LP
        prob, x_vars, y_vars = self._build_ride_hailing_lp(
            clients, taxis, edges, price_grid, acceptance_prob, cost)
        
        # Solve the LP
        if self.solver_name == 'PULP_CBC_CMD':
            solver = pl.PULP_CBC_CMD(msg=self.solver_verbose, timeLimit=self.solver_timeout)
        else:
            solver = None
            
        prob.solve(solver)
        
        # Extract solution
        if prob.status == pl.LpStatusOptimal:
            prices, acceptance_probs = self._extract_solution(
                clients, price_grid, x_vars, y_vars, acceptance_prob)
        else:
            # Fallback to simple pricing if LP fails
            prices = np.array([np.mean(price_grid[c]) for c in clients])
            acceptance_probs = np.array([acceptance_prob[(c, prices[c])] 
                                       if (c, prices[c]) in acceptance_prob 
                                       else 0.5 for c in clients])
        
        # Calculate edge weights for evaluation
        w_matrix = self._calculate_edge_weights(distance_matrix, trip_distances, 18.0, 25.0)
        
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
                'algorithm': 'gupta_nagarajan_linear_program',
                'lp_status': prob.status,
                'lp_objective_value': pl.value(prob.objective) if prob.status == pl.LpStatusOptimal else None,
                'acceptance_function': self.acceptance_function,
                'n_requesters': n,
                'n_taxis': m,
                'n_edges': len(edges)
            }
        )
    
    def _create_price_grids(self, clients: List[int], trip_amounts: np.ndarray, 
                           trip_distances: np.ndarray) -> Dict[int, List[float]]:
        """
        Create discrete price grids for each client.
        
        The price grid Π_c is a discrete menu of candidate prices for client c.
        """
        price_grid = {}
        
        for c in clients:
            # Base price from trip characteristics
            base_price = trip_amounts[c] / trip_distances[c] if trip_distances[c] > 0 else 10.0
            
            # Create price grid around base price
            min_price = base_price * self.min_price_factor
            max_price = base_price * self.max_price_factor
            
            prices = np.linspace(min_price, max_price, self.price_grid_size)
            price_grid[c] = prices.tolist()
        
        return price_grid
    
    def _calculate_acceptance_probabilities(self, clients: List[int], 
                                          price_grid: Dict[int, List[float]],
                                          trip_amounts: np.ndarray) -> Dict[Tuple[int, float], float]:
        """
        Calculate acceptance probabilities p_c(π) for each client and price.
        
        This multiplies y in constraint (2) of the LP.
        """
        acceptance_prob = {}
        
        for c in clients:
            for pi in price_grid[c]:
                if self.acceptance_function == 'sigmoid':
                    # Sigmoid acceptance function
                    q_u = trip_amounts[c]  # reservation price
                    if abs(q_u) < 1e-6:
                        prob = 0.5
                    else:
                        exponent = -(pi - self.sigmoid_beta * q_u) / (self.sigmoid_gamma * abs(q_u))
                        exponent = np.clip(exponent, -50, 50)
                        prob = 1 - 1 / (1 + np.exp(exponent))
                else:
                    # Piecewise linear (simplified)
                    q_u = trip_amounts[c]
                    if pi < q_u:
                        prob = 1.0
                    elif pi <= 1.5 * q_u:
                        prob = (-1 / (0.5 * q_u)) * pi + 1.5 / 0.5
                    else:
                        prob = 0.0
                
                # Ensure probability is in [0, 1]
                prob = max(0.0, min(1.0, prob))
                acceptance_prob[(c, pi)] = prob
        
        return acceptance_prob
    
    def _calculate_costs(self, edges: set, distance_matrix: np.ndarray, 
                        trip_distances: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Calculate driving costs w_{c,t} for each feasible edge.
        
        This appears in the objective function of the LP.
        """
        cost = {}
        alpha = 18.0  # Opportunity cost parameter
        s_taxi = 25.0  # Taxi speed
        
        for (c, t) in edges:
            # Cost = opportunity cost of distance and trip
            driving_cost = (distance_matrix[c, t] + trip_distances[c]) / s_taxi * alpha
            cost[(c, t)] = driving_cost
        
        return cost
    
    def _build_ride_hailing_lp(self, clients: List[int], taxis: List[int], 
                              edges: set, price_grid: Dict[int, List[float]],
                              acceptance_prob: Dict[Tuple[int, float], float],
                              cost: Dict[Tuple[int, int], float]) -> Tuple[pl.LpProblem, Dict, Dict]:
        """
        Build the Gupta-Nagarajan LP for ride-hailing revenue maximization.
        
        This is the exact implementation from the provided PuLP code.
        """
        # Create the model
        prob = pl.LpProblem("RideHailing_GN_LP", pl.LpMaximize)
        
        # Variables
        # y_{c,π} = probability we offer price π to rider c
        y = {}
        for c in clients:
            for pi in price_grid[c]:
                y[(c, pi)] = pl.LpVariable(f"y_{c}_{pi}", lowBound=0, upBound=1)
        
        # x_{c,t,π} = prob. rider c accepts π and is matched to taxi t
        x = {}
        for (c, t) in edges:
            for pi in price_grid[c]:
                x[(c, t, pi)] = pl.LpVariable(f"x_{c}_{t}_{pi}", lowBound=0, upBound=1)
        
        # Objective: maximize total expected profit
        prob += pl.lpSum(
            (pi - cost[(c, t)]) * x[(c, t, pi)]
            for (c, t) in edges
            for pi in price_grid[c]
        ), "Total_expected_profit"
        
        # Constraints
        
        # (1) Probe at most one price per rider
        for c in clients:
            prob += (
                pl.lpSum(y[(c, pi)] for pi in price_grid[c]) <= 1,
                f"Offer_once_{c}"
            )
        
        # (2) Matching only after acceptance
        for (c, t) in edges:
            for pi in price_grid[c]:
                prob += (
                    x[(c, t, pi)] <= acceptance_prob[(c, pi)] * y[(c, pi)],
                    f"Link_{c}_{t}_{pi}"
                )
        
        # (3) Taxi capacity: one rider per taxi
        for t in taxis:
            prob += (
                pl.lpSum(
                    x[(c, t, pi)]
                    for c in clients if (c, t) in edges
                    for pi in price_grid[c]
                ) <= 1,
                f"Taxi_cap_{t}"
            )
        
        return prob, x, y
    
    def _extract_solution(self, clients: List[int], price_grid: Dict[int, List[float]],
                         x_vars: Dict, y_vars: Dict, 
                         acceptance_prob: Dict[Tuple[int, float], float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract prices and acceptance probabilities from LP solution.
        """
        n = len(clients)
        prices = np.zeros(n)
        acceptance_probs = np.zeros(n)
        
        for c in clients:
            # Find the price with highest y value (most likely to be offered)
            best_pi = None
            best_y_val = 0.0
            
            for pi in price_grid[c]:
                y_val = y_vars[(c, pi)].varValue or 0.0
                if y_val > best_y_val:
                    best_y_val = y_val
                    best_pi = pi
            
            if best_pi is not None:
                prices[c] = best_pi
                acceptance_probs[c] = acceptance_prob.get((c, best_pi), 0.0)
            else:
                # Fallback
                prices[c] = np.mean(price_grid[c])
                acceptance_probs[c] = 0.5
        
        return prices, acceptance_probs
    
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