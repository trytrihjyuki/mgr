"""
Linear Programming approach for ride-hailing pricing optimization.
Based on Gupta-Nagarajan reduction of the Myerson revenue maximization problem.
"""

import pulp
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import time

# Type aliases for clarity
Client = int
Taxi = int  
Price = float
Cost = float
Prob = float

logger = logging.getLogger(__name__)

class LPPricingOptimizer:
    """
    Linear Programming optimizer for ride-hailing pricing using Gupta-Nagarajan reduction.
    """
    
    def __init__(self, solver_name='CBC', verbose=False):
        """
        Initialize the LP optimizer.
        
        Args:
            solver_name: Name of the solver to use ('CBC', 'GUROBI', 'CPLEX', etc.)
            verbose: Whether to enable verbose solver output
        """
        self.solver_name = solver_name
        self.verbose = verbose
        self.last_solve_time = 0
        self.last_objective_value = 0
        
    def _get_solver(self):
        """Get the appropriate PuLP solver."""
        if self.solver_name.upper() == 'CBC':
            return pulp.PULP_CBC_CMD(msg=self.verbose)
        elif self.solver_name.upper() == 'GUROBI':
            return pulp.GUROBI_CMD(msg=self.verbose)
        elif self.solver_name.upper() == 'CPLEX':
            return pulp.CPLEX_CMD(msg=self.verbose)
        else:
            return pulp.PULP_CBC_CMD(msg=self.verbose)  # Default fallback
    
    def solve_pricing_lp(self, 
                        clients: List[Client],
                        taxis: List[Taxi], 
                        edges: Dict[Tuple[Client, Taxi], Cost],
                        price_grid: Dict[Client, List[Price]],
                        acceptance_prob: Dict[Tuple[Client, Price], Prob]) -> Dict[str, Any]:
        """
        Solve the Gupta-Nagarajan LP for ride-hailing pricing.
        
        Args:
            clients: List of client/rider IDs
            taxis: List of taxi IDs  
            edges: Dictionary mapping (client, taxi) pairs to travel costs
            price_grid: Dictionary mapping clients to lists of candidate prices
            acceptance_prob: Dictionary mapping (client, price) to acceptance probability
            
        Returns:
            Dictionary containing:
                - 'prices': Optimal prices for each client
                - 'objective_value': Optimal expected profit
                - 'solve_time': Time taken to solve
                - 'status': Solver status
                - 'y_values': Pricing decision variables
                - 'x_values': Matching decision variables
        """
        start_time = time.time()
        
        try:
            # Build the LP problem
            prob, y_vars, x_vars = self._build_lp(
                clients, taxis, edges, price_grid, acceptance_prob
            )
            
            # Solve the problem
            solver = self._get_solver()
            status = prob.solve(solver)
            
            solve_time = time.time() - start_time
            self.last_solve_time = solve_time
            
            if status == pulp.LpStatusOptimal:
                # Extract solution
                objective_value = pulp.value(prob.objective)
                self.last_objective_value = objective_value
                
                # Extract pricing decisions
                prices = {}
                y_values = {}
                for (c, pi), var in y_vars.items():
                    y_val = var.varValue if var.varValue is not None else 0
                    y_values[(c, pi)] = y_val
                    if y_val > 1e-6:  # Only consider significant probabilities
                        if c not in prices:
                            prices[c] = []
                        prices[c].append((pi, y_val))
                
                # Extract matching decisions  
                x_values = {}
                for (c, t, pi), var in x_vars.items():
                    x_val = var.varValue if var.varValue is not None else 0
                    x_values[(c, t, pi)] = x_val
                
                logger.info(f"LP solved optimally in {solve_time:.3f}s with objective {objective_value:.4f}")
                
                return {
                    'prices': prices,
                    'objective_value': objective_value,
                    'solve_time': solve_time,
                    'status': 'optimal',
                    'y_values': y_values,
                    'x_values': x_values
                }
                
            else:
                logger.warning(f"LP solver failed with status: {pulp.LpStatus[status]}")
                return {
                    'prices': {},
                    'objective_value': 0,
                    'solve_time': solve_time,
                    'status': pulp.LpStatus[status],
                    'y_values': {},
                    'x_values': {}
                }
                
        except Exception as e:
            logger.error(f"Error solving LP: {str(e)}")
            return {
                'prices': {},
                'objective_value': 0,
                'solve_time': time.time() - start_time,
                'status': 'error',
                'y_values': {},
                'x_values': {},
                'error': str(e)
            }
    
    def _build_lp(self, 
                  clients: List[Client],
                  taxis: List[Taxi],
                  edges: Dict[Tuple[Client, Taxi], Cost],
                  price_grid: Dict[Client, List[Price]],
                  acceptance_prob: Dict[Tuple[Client, Price], Prob]) -> Tuple[pulp.LpProblem, Dict, Dict]:
        """
        Build the Gupta-Nagarajan linear program.
        
        Returns:
            Tuple of (problem, y_variables, x_variables)
        """
        
        prob = pulp.LpProblem('RideHailing_GN_LP', pulp.LpMaximize)
        
        # Decision variables
        # y[c, pi] = probability of offering price pi to client c
        y_vars = {}
        for c in clients:
            for pi in price_grid.get(c, []):
                var_name = f"y_{c}_{pi}"
                y_vars[(c, pi)] = pulp.LpVariable(var_name, lowBound=0.0, upBound=1.0, cat='Continuous')
        
        # x[c, t, pi] = probability client c accepts price pi and is matched to taxi t
        x_vars = {}
        for c in clients:
            for pi in price_grid.get(c, []):
                for t in taxis:
                    if (c, t) in edges:
                        var_name = f"x_{c}_{t}_{pi}"
                        x_vars[(c, t, pi)] = pulp.LpVariable(var_name, lowBound=0.0, upBound=1.0, cat='Continuous')
        
        # Objective: maximize expected profit
        revenue_terms = []
        for c in clients:
            for pi in price_grid.get(c, []):
                for t in taxis:
                    if (c, t) in edges and (c, t, pi) in x_vars:
                        profit = pi - edges[(c, t)]
                        revenue_terms.append(profit * x_vars[(c, t, pi)])
        
        prob += pulp.lpSum(revenue_terms), 'Total_Expected_Profit'
        
        # Constraints
        
        # (1) Offer at most one price to each client
        for c in clients:
            if c in price_grid and price_grid[c]:
                prob += (
                    pulp.lpSum(y_vars[(c, pi)] for pi in price_grid[c] if (c, pi) in y_vars) <= 1,
                    f'OnePrice_{c}'
                )
        
        # (2) Matching limited by acceptance probability  
        for c in clients:
            for pi in price_grid.get(c, []):
                if (c, pi) in y_vars and (c, pi) in acceptance_prob:
                    matching_sum = pulp.lpSum(
                        x_vars[(c, t, pi)] 
                        for t in taxis 
                        if (c, t) in edges and (c, t, pi) in x_vars
                    )
                    prob += (
                        matching_sum <= acceptance_prob[(c, pi)] * y_vars[(c, pi)],
                        f'AcceptanceLimit_{c}_{pi}'
                    )
        
        # (3) Each taxi serves at most one client
        for t in taxis:
            taxi_assignments = []
            for c in clients:
                for pi in price_grid.get(c, []):
                    if (c, t) in edges and (c, t, pi) in x_vars:
                        taxi_assignments.append(x_vars[(c, t, pi)])
            
            if taxi_assignments:
                prob += (
                    pulp.lpSum(taxi_assignments) <= 1,
                    f'TaxiCapacity_{t}'
                )
        
        return prob, y_vars, x_vars
    
    def extract_deterministic_prices(self, solution: Dict[str, Any]) -> Dict[Client, Price]:
        """
        Extract deterministic pricing decisions from LP solution.
        For clients with fractional solutions, select the price with highest probability.
        
        Args:
            solution: Solution dictionary from solve_pricing_lp
            
        Returns:
            Dictionary mapping client to selected price
        """
        deterministic_prices = {}
        
        if 'prices' not in solution:
            return deterministic_prices
            
        for client, price_probs in solution['prices'].items():
            if price_probs:
                # Select price with highest probability
                best_price, best_prob = max(price_probs, key=lambda x: x[1])
                deterministic_prices[client] = best_price
                
        return deterministic_prices

def create_price_grid(clients: List[Client], 
                     base_prices: List[Price], 
                     client_multipliers: Dict[Client, float] = None) -> Dict[Client, List[Price]]:
    """
    Create price grids for clients.
    
    Args:
        clients: List of client IDs
        base_prices: Base price levels  
        client_multipliers: Optional multipliers for each client
        
    Returns:
        Dictionary mapping clients to their price grids
    """
    price_grid = {}
    
    for c in clients:
        multiplier = client_multipliers.get(c, 1.0) if client_multipliers else 1.0
        price_grid[c] = [p * multiplier for p in base_prices]
        
    return price_grid 