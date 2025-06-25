#!/usr/bin/env python3
"""
Linear Programming Method Implementation
Based on Gupta-Nagarajan formulation for ride-hailing revenue maximization
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import pulp as pl
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    logger.warning("PuLP not available - LinearProgram method will use simplified implementation")


@dataclass
class LinearProgramResult:
    """Result structure for Linear Programming method"""
    method_name: str = "LinearProgram"
    prices: np.ndarray = None
    objective_value: float = 0.0
    computation_time: float = 0.0
    solver_status: str = "Unknown"
    y_variables: Dict[Tuple[int, float], float] = None  # (client, price) -> probability
    x_variables: Dict[Tuple[int, int, float], float] = None  # (client, taxi, price) -> probability
    price_grid: Dict[int, List[float]] = None
    acceptance_probabilities: np.ndarray = None
    matched_pairs: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.matched_pairs is None:
            self.matched_pairs = []
        if self.y_variables is None:
            self.y_variables = {}
        if self.x_variables is None:
            self.x_variables = {}
        if self.price_grid is None:
            self.price_grid = {}


class LinearProgramMethod:
    """
    Implementation of Linear Programming method for taxi pricing
    Based on Gupta-Nagarajan formulation from the theoretical framework
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Linear Programming method with configuration
        
        Args:
            config: Configuration dictionary containing method parameters
        """
        self.min_price_factor = config.get('min_price_factor', 0.5)
        self.max_price_factor = config.get('max_price_factor', 2.0)
        self.price_grid_size = config.get('price_grid_size', 10)
        self.solver_name = config.get('solver_name', 'PULP_CBC_CMD')
        self.solver_timeout = config.get('solver_timeout', 300)
        self.solver_verbose = config.get('solver_verbose', False)
        self.acceptance_model = config.get('acceptance_model', 'sigmoid')
        
        # Acceptance function parameters
        self.sigmoid_params = config.get('sigmoid_params', {
            'beta': 1.3,
            'gamma': 0.3 * math.sqrt(3) / math.pi
        })
        
        logger.info(f"Initialized LinearProgram method with grid_size={self.price_grid_size}")
    
    def build_price_grid(self, clients_data: np.ndarray) -> Dict[int, List[float]]:
        """
        Build discrete price grid for each client
        Following Gupta-Nagarajan approach: discrete menu of posted prices
        
        Args:
            clients_data: Array of client data
            
        Returns:
            Dictionary mapping client_id -> list of candidate prices
        """
        price_grid = {}
        
        for c in range(len(clients_data)):
            client_data = clients_data[c]
            
            # Extract base price information
            trip_distance = float(client_data[2]) if len(client_data) > 2 else 1.0
            total_amount = float(client_data[3]) if len(client_data) > 3 else 10.0
            
            # Calculate base price per km
            base_price_per_km = total_amount / max(trip_distance, 0.1)
            
            # Create price grid around the base price
            min_price = base_price_per_km * self.min_price_factor
            max_price = base_price_per_km * self.max_price_factor
            
            # Generate logarithmically spaced price grid
            if self.price_grid_size == 1:
                prices = [base_price_per_km]
            else:
                prices = np.logspace(
                    math.log10(min_price), 
                    math.log10(max_price), 
                    self.price_grid_size
                ).tolist()
            
            price_grid[c] = prices
        
        return price_grid
    
    def calculate_acceptance_probability(self, client_id: int, price: float, 
                                      clients_data: np.ndarray) -> float:
        """
        Calculate acceptance probability p_c(π) for client c at price π
        
        Args:
            client_id: Client identifier
            price: Proposed price
            clients_data: Client data array
            
        Returns:
            Acceptance probability p_c(π)
        """
        client_data = clients_data[client_id]
        total_amount = float(client_data[3]) if len(client_data) > 3 else 10.0
        
        if self.acceptance_model == 'linear':
            # Linear model: p_c(π) = max(0, min(1, a - b*π))
            a, b = 1.5, 0.1  # Parameters
            return max(0.0, min(1.0, a - b * price))
        else:
            # Sigmoid model (default): following exact formulation
            beta = self.sigmoid_params['beta']
            gamma = self.sigmoid_params['gamma']
            
            if abs(total_amount) < 1e-6:
                return 0.5
            else:
                exponent = -(price - beta * total_amount) / (gamma * abs(total_amount))
                return 1 - 1 / (1 + math.exp(max(-50, min(50, exponent))))
    
    def calculate_cost(self, client_id: int, taxi_id: int, 
                      clients_data: np.ndarray, taxis_data: np.ndarray) -> float:
        """
        Calculate driving cost w_{c,t} between client c and taxi t
        
        Args:
            client_id: Client identifier
            taxi_id: Taxi identifier
            clients_data: Client data array
            taxis_data: Taxi data array
            
        Returns:
            Driving cost w_{c,t}
        """
        # Simplified cost calculation (in practice, use actual distance/time)
        # Cost includes distance and opportunity cost
        trip_distance = float(clients_data[client_id][2]) if len(clients_data[client_id]) > 2 else 1.0
        base_cost = 2.0  # Base operating cost
        distance_cost = trip_distance * 0.5  # Cost per km
        
        return base_cost + distance_cost
    
    def solve_with_pulp(self, clients_data: np.ndarray, taxis_data: np.ndarray) -> LinearProgramResult:
        """
        Solve using PuLP linear programming solver
        Implements exact Gupta-Nagarajan LP formulation
        
        Args:
            clients_data: Array of client data
            taxis_data: Array of taxi data
            
        Returns:
            LinearProgramResult containing solution
        """
        import time
        start_time = time.time()
        
        n_clients = len(clients_data)
        n_taxis = len(taxis_data)
        
        logger.info(f"Building Gupta-Nagarajan LP with {n_clients} clients, {n_taxis} taxis")
        
        # Build price grid for each client
        price_grid = self.build_price_grid(clients_data)
        
        # Define feasible edges (client-taxi pairs)
        edges = [(c, t) for c in range(n_clients) for t in range(n_taxis)]
        
        # Calculate acceptance probabilities and costs
        acceptance_prob = {}
        cost = {}
        
        for c in range(n_clients):
            for pi in price_grid[c]:
                acceptance_prob[(c, pi)] = self.calculate_acceptance_probability(c, pi, clients_data)
        
        for (c, t) in edges:
            cost[(c, t)] = self.calculate_cost(c, t, clients_data, taxis_data)
        
        # Build Linear Program (exact Gupta-Nagarajan formulation)
        prob = pl.LpProblem("RideHailing_GN_LP", pl.LpMaximize)
        
        # Variables
        # y_{c,π} = probability we offer price π to client c
        y = {(c, pi): pl.LpVariable(f"y_{c}_{pi}", lowBound=0, upBound=1)
             for c in range(n_clients) for pi in price_grid[c]}
        
        # x_{c,t,π} = probability client c accepts π and is matched to taxi t
        x = {(c, t, pi): pl.LpVariable(f"x_{c}_{t}_{pi}", lowBound=0, upBound=1)
             for (c, t) in edges for pi in price_grid[c]}
        
        # Objective: maximize expected profit (Equation in paper)
        prob += pl.lpSum(
            (pi - cost[(c, t)]) * x[(c, t, pi)]
            for (c, t) in edges
            for pi in price_grid[c]
        ), "Total_expected_profit"
        
        # Constraints
        
        # (1) Probe at most one price per client
        for c in range(n_clients):
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
        
        # (3) Taxi capacity: one client per taxi
        for t in range(n_taxis):
            prob += (
                pl.lpSum(
                    x[(c, t, pi)]
                    for c in range(n_clients) if (c, t) in edges
                    for pi in price_grid[c]
                ) <= 1,
                f"Taxi_cap_{t}"
            )
        
        # Solve the LP
        if self.solver_verbose:
            solver = pl.PULP_CBC_CMD(msg=1, timeLimit=self.solver_timeout)
        else:
            solver = pl.PULP_CBC_CMD(msg=0, timeLimit=self.solver_timeout)
        
        prob.solve(solver)
        
        # Extract solution
        solver_status = pl.LpStatus[prob.status]
        objective_value = pl.value(prob.objective) if prob.status == pl.LpStatusOptimal else 0.0
        
        # Extract variable values
        y_values = {}
        x_values = {}
        
        for (c, pi), var in y.items():
            if var.varValue and var.varValue > 1e-6:
                y_values[(c, pi)] = var.varValue
        
        for (c, t, pi), var in x.items():
            if var.varValue and var.varValue > 1e-6:
                x_values[(c, t, pi)] = var.varValue
        
        # Calculate final prices and create solution
        prices = np.zeros(n_clients)
        acceptance_probs = np.zeros(n_clients)
        matched_pairs = []
        
        for c in range(n_clients):
            # Find selected price for this client
            selected_price = 0.0
            max_prob = 0.0
            
            for pi in price_grid[c]:
                if (c, pi) in y_values and y_values[(c, pi)] > max_prob:
                    max_prob = y_values[(c, pi)]
                    selected_price = pi
            
            prices[c] = selected_price
            if selected_price > 0:
                acceptance_probs[c] = acceptance_prob[(c, selected_price)]
            
            # Find matching for this client
            for t in range(n_taxis):
                for pi in price_grid[c]:
                    if (c, t, pi) in x_values and x_values[(c, t, pi)] > 1e-6:
                        matched_pairs.append((c, t))
                        break
        
        computation_time = time.time() - start_time
        
        logger.info(f"LP solved in {computation_time:.3f}s, status: {solver_status}, objective: {objective_value:.2f}")
        
        return LinearProgramResult(
            prices=prices,
            objective_value=objective_value,
            computation_time=computation_time,
            solver_status=solver_status,
            y_variables=y_values,
            x_variables=x_values,
            price_grid=price_grid,
            acceptance_probabilities=acceptance_probs,
            matched_pairs=matched_pairs
        )
    
    def solve_simplified(self, clients_data: np.ndarray, taxis_data: np.ndarray) -> LinearProgramResult:
        """
        Simplified implementation when PuLP is not available
        Uses greedy heuristic approximation
        
        Args:
            clients_data: Array of client data
            taxis_data: Array of taxi data
            
        Returns:
            LinearProgramResult containing approximate solution
        """
        import time
        start_time = time.time()
        
        n_clients = len(clients_data)
        n_taxis = len(taxis_data)
        
        logger.info(f"Using simplified LP approximation with {n_clients} clients, {n_taxis} taxis")
        
        # Build price grid
        price_grid = self.build_price_grid(clients_data)
        
        # Greedy assignment
        prices = np.zeros(n_clients)
        acceptance_probs = np.zeros(n_clients)
        matched_pairs = []
        
        available_taxis = set(range(n_taxis))
        
        for c in range(n_clients):
            if not available_taxis:
                break
            
            # Find best price for this client
            best_profit = 0.0
            best_price = 0.0
            best_taxi = None
            
            for pi in price_grid[c]:
                acceptance_prob = self.calculate_acceptance_probability(c, pi, clients_data)
                
                for t in available_taxis:
                    cost = self.calculate_cost(c, t, clients_data, taxis_data)
                    expected_profit = acceptance_prob * (pi - cost)
                    
                    if expected_profit > best_profit:
                        best_profit = expected_profit
                        best_price = pi
                        best_taxi = t
            
            if best_taxi is not None:
                prices[c] = best_price
                acceptance_probs[c] = self.calculate_acceptance_probability(c, best_price, clients_data)
                matched_pairs.append((c, best_taxi))
                available_taxis.remove(best_taxi)
        
        objective_value = sum(prices[c] for c, _ in matched_pairs)
        computation_time = time.time() - start_time
        
        logger.info(f"Simplified LP completed in {computation_time:.3f}s, objective: {objective_value:.2f}")
        
        return LinearProgramResult(
            prices=prices,
            objective_value=objective_value,
            computation_time=computation_time,
            solver_status="Approximate",
            price_grid=price_grid,
            acceptance_probabilities=acceptance_probs,
            matched_pairs=matched_pairs
        )
    
    def solve(self, requesters_data: np.ndarray, taxis_data: np.ndarray) -> LinearProgramResult:
        """
        Main solve method for Linear Programming pricing optimization
        
        Args:
            requesters_data: Array of requester data
            taxis_data: Array of taxi data
            
        Returns:
            LinearProgramResult containing solution
        """
        if PULP_AVAILABLE:
            return self.solve_with_pulp(requesters_data, taxis_data)
        else:
            return self.solve_simplified(requesters_data, taxis_data) 