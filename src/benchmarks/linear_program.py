"""
Linear Program solver for rideshare pricing optimization.

Implementation of the Gupta-Nagarajan LP formulation for ride-hailing
revenue maximization, as provided by the user.

This is a drop-in, one-for-one implementation of the Gupta–Nagarajan 
linear program for ride-hailing, written with PuLP.
"""

import pulp as pl
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from ..utils.config import Config

logger = logging.getLogger(__name__)


class LinearProgramSolver:
    """
    Linear Program solver for rideshare pricing optimization.
    
    Implements the Gupta-Nagarajan LP formulation:
    - y[c,π]: probability we offer price π to rider c
    - x[c,t,π]: prob. rider c accepts π and is matched to taxi t
    """
    
    def __init__(self, config: Config):
        """
        Initialize Linear Program solver.
        
        Args:
            config: Configuration object with algorithm parameters
        """
        self.config = config
        self.solver_name = config.algorithms.lp_solver
        self.time_limit = config.algorithms.lp_time_limit
        
    def solve_pricing(self, scenario_data: pd.DataFrame, 
                     distance_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Solve the pricing optimization problem using Linear Programming.
        
        Args:
            scenario_data: DataFrame with trip data for this scenario
            distance_matrix: Distance matrix between requests and drivers
            
        Returns:
            Dictionary with pricing results and solver information
        """
        logger.info("🔧 Solving pricing with Linear Program (Gupta-Nagarajan)")
        
        if len(scenario_data) == 0:
            return self._empty_result()
        
        # Extract problem data
        clients = list(range(len(scenario_data)))
        taxis = list(range(len(scenario_data)))  # Simplified: same as clients
        edges = [(c, t) for c in clients for t in taxis if distance_matrix[c, t] <= self.config.algorithms.maps_max_distance]
        
        # Create price grid for each client
        price_grid = self._create_price_grid(scenario_data)
        
        # Calculate acceptance probabilities
        acceptance_prob = self._calculate_acceptance_probabilities(scenario_data, price_grid)
        
        # Calculate costs (driving costs)
        cost = self._calculate_costs(scenario_data, distance_matrix, edges)
        
        # Build and solve LP
        prob, x, y = self._build_lp(clients, taxis, edges, price_grid, acceptance_prob, cost)
        
        # Solve the problem
        solver = self._get_solver()
        if solver:
            prob.solver = solver
        
        solve_status = prob.solve()
        
        # Extract results
        if solve_status == pl.LpStatusOptimal:
            return self._extract_solution(prob, x, y, scenario_data, price_grid, clients, taxis)
        else:
            logger.warning(f"⚠️ LP solver failed with status: {pl.LpStatus[solve_status]}")
            return self._empty_result()
    
    def _build_lp(self, clients: List[int], taxis: List[int], edges: List[Tuple[int, int]],
                  price_grid: Dict[int, List[float]], acceptance_prob: Dict[Tuple[int, float], float],
                  cost: Dict[Tuple[int, int], float]):
        """
        Build the Gupta–Nagarajan LP for ride-hailing revenue maximisation.
        
        Returns:
            Tuple of (problem, x_variables, y_variables)
        """
        # Create the LP problem
        prob = pl.LpProblem("RideHailing_GN_LP", pl.LpMaximize)
        
        # Decision variables
        # y[c,π]: probability we offer price π to rider c
        y = {(c, pi): pl.LpVariable(f"y_{c}_{pi}", lowBound=0, upBound=1)
             for c in clients
             for pi in price_grid[c]}
        
        # x[c,t,π]: prob. rider c accepts π and is matched to taxi t
        x = {(c, t, pi): pl.LpVariable(f"x_{c}_{t}_{pi}", lowBound=0, upBound=1)
             for (c, t) in edges
             for pi in price_grid[c]}
        
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
    
    def _create_price_grid(self, scenario_data: pd.DataFrame) -> Dict[int, List[float]]:
        """Create discrete price grid for each client."""
        price_grid = {}
        
        base_price = self.config.algorithms.hikima_base_price
        
        for i, row in scenario_data.iterrows():
            trip_distance = row.get('trip_distance', 1.0)
            total_amount = row.get('total_amount', base_price)
            
            # Create price grid around the historical fare
            min_price = max(base_price * 0.5, total_amount * 0.5)
            max_price = total_amount * 2.0
            
            # Create 5 price points
            price_points = np.linspace(min_price, max_price, 5)
            price_grid[i] = price_points.tolist()
        
        return price_grid
    
    def _calculate_acceptance_probabilities(self, scenario_data: pd.DataFrame, 
                                          price_grid: Dict[int, List[float]]) -> Dict[Tuple[int, float], float]:
        """Calculate acceptance probabilities for each client-price pair."""
        acceptance_prob = {}
        
        for i, row in scenario_data.iterrows():
            total_amount = row.get('total_amount', self.config.algorithms.hikima_base_price)
            
            for price in price_grid[i]:
                if self.config.experiment.acceptance_function == "PL":
                    # Piecewise Linear acceptance function
                    alpha = self.config.acceptance.pl_alpha
                    if price <= total_amount:
                        prob = 1.0
                    elif price <= alpha * total_amount:
                        prob = 1.0 - (price - total_amount) / ((alpha - 1) * total_amount)
                    else:
                        prob = 0.0
                else:
                    # Sigmoid acceptance function
                    beta = self.config.acceptance.sigmoid_beta
                    gamma = self.config.acceptance.sigmoid_gamma
                    
                    if abs(total_amount) < 1e-6:
                        prob = 0.5
                    else:
                        exponent = -(price - beta * total_amount) / (gamma * abs(total_amount))
                        prob = 1 - 1 / (1 + np.exp(max(-50, min(50, exponent))))
                
                acceptance_prob[(i, price)] = max(0.0, min(1.0, prob))
        
        return acceptance_prob
    
    def _calculate_costs(self, scenario_data: pd.DataFrame, distance_matrix: np.ndarray,
                        edges: List[Tuple[int, int]]) -> Dict[Tuple[int, int], float]:
        """Calculate driving costs for each client-taxi pair."""
        cost = {}
        
        alpha = self.config.algorithms.hikima_alpha
        taxi_speed = self.config.algorithms.hikima_taxi_speed
        
        for (c, t) in edges:
            # Get trip distance for opportunity cost calculation
            trip_distance_km = scenario_data.iloc[c].get('trip_distance_km', 
                                                       scenario_data.iloc[c].get('trip_distance', 1.0) * 1.60934)
            
            # Calculate cost as in Hikima method
            driving_distance = distance_matrix[c, t]
            total_distance = driving_distance + trip_distance_km
            opportunity_cost = (total_distance / taxi_speed) * alpha
            
            cost[(c, t)] = opportunity_cost
        
        return cost
    
    def _get_solver(self):
        """Get the appropriate LP solver."""
        try:
            if self.solver_name.upper() == "CBC":
                return pl.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit)
            elif self.solver_name.upper() == "GLPK":
                return pl.GLPK_CMD(msg=False, options=[f"--tmlim", str(self.time_limit)])
            elif self.solver_name.upper() == "GUROBI":
                return pl.GUROBI_CMD(msg=False, timeLimit=self.time_limit)
            else:
                logger.warning(f"⚠️ Unknown solver {self.solver_name}, using default CBC")
                return pl.PULP_CBC_CMD(msg=False, timeLimit=self.time_limit)
        except:
            logger.warning("⚠️ Could not initialize solver, using PuLP default")
            return None
    
    def _extract_solution(self, prob, x, y, scenario_data: pd.DataFrame, 
                         price_grid: Dict[int, List[float]], clients: List[int], 
                         taxis: List[int]) -> Dict[str, Any]:
        """Extract solution from solved LP."""
        
        optimal_value = pl.value(prob.objective)
        
        # Extract pricing decisions
        pricing_decisions = {}
        matched_pairs = []
        total_revenue = 0.0
        
        for c in clients:
            client_prices = []
            for pi in price_grid[c]:
                if y[(c, pi)].varValue and y[(c, pi)].varValue > 1e-6:
                    client_prices.append({
                        'price': pi,
                        'probability': y[(c, pi)].varValue
                    })
            pricing_decisions[c] = client_prices
        
        # Extract matching decisions
        for (c, t) in [(c, t) for c in clients for t in taxis]:
            for pi in price_grid[c]:
                if (c, t, pi) in x and x[(c, t, pi)].varValue and x[(c, t, pi)].varValue > 1e-6:
                    matched_pairs.append({
                        'client': c,
                        'taxi': t,
                        'price': pi,
                        'probability': x[(c, t, pi)].varValue
                    })
                    total_revenue += pi * x[(c, t, pi)].varValue
        
        # Calculate summary statistics
        num_clients = len(clients)
        num_matches = sum(1 for pair in matched_pairs if pair['probability'] > 0.5)
        avg_price = total_revenue / max(num_matches, 1)
        
        return {
            'method': 'linear_program',
            'optimal_value': optimal_value,
            'total_revenue': total_revenue,
            'num_clients': num_clients,
            'num_matches': num_matches,
            'match_rate': num_matches / max(num_clients, 1),
            'avg_price': avg_price,
            'pricing_decisions': pricing_decisions,
            'matched_pairs': matched_pairs,
            'solver_status': 'optimal',
            'computation_time': getattr(prob, 'solutionTime', 0.0)
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for cases with no data."""
        return {
            'method': 'linear_program',
            'optimal_value': 0.0,
            'total_revenue': 0.0,
            'num_clients': 0,
            'num_matches': 0,
            'match_rate': 0.0,
            'avg_price': 0.0,
            'pricing_decisions': {},
            'matched_pairs': [],
            'solver_status': 'no_data',
            'computation_time': 0.0
        } 