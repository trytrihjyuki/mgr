"""MinMaxCostFlow pricing method implementation (Hikima et al. main method).

This implements the min-cost flow algorithm with delta-scaling from:
"Dynamic Pricing in Ride-Hailing Platforms: A Reinforcement Learning Approach"
by Hikima et al.
"""

import numpy as np
import networkx as nx
import math
from typing import Dict, Any, List, Tuple
from .base import BasePricingMethod


class MinMaxCostFlowMethod(BasePricingMethod):
    """
    Min-Max Cost Flow pricing method using Hikima et al.'s approach.
    
    This method solves the pricing problem using a min-cost flow algorithm
    with delta-scaling for efficiency.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MinMaxCostFlow pricing method."""
        super().__init__(config)
        
        # Delta-scaling parameters
        self.delta_threshold = config.get('delta_threshold', 0.001)
        self.initial_delta_multiplier = config.get('initial_delta_mult', 1.0)
        
    def get_method_name(self) -> str:
        """Get method name."""
        return "MinMaxCostFlow"
    
    def compute_prices(self, scenario_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute prices using the min-cost flow algorithm with delta-scaling.
        
        Args:
            scenario_data: Dictionary with scenario data
            
        Returns:
            Array of prices for each requester
        """
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        edge_weights = scenario_data['edge_weights']
        trip_amounts = scenario_data['trip_amounts']
        
        # Handle edge cases
        if n_requesters == 0:
            return np.array([])
        if n_taxis == 0:
            return np.zeros(n_requesters)
        
        # We'll compute prices that work reasonably well for both acceptance functions
        # Use PL coefficients as default since the method optimizes the flow
        c = 2.0 / trip_amounts  # coefficient for PL function
        d = 3.0  # constant for PL function
        
        # Build and solve min-cost flow problem
        prices = self._solve_min_cost_flow(
            n_requesters, n_taxis, edge_weights, c, d, 'PL', trip_amounts
        )
        
        return prices
    
    def _solve_min_cost_flow(
        self, 
        n: int, 
        m: int, 
        W: np.ndarray,
        c: np.ndarray,
        d: float,
        acceptance_func: str,
        trip_amounts: np.ndarray
    ) -> np.ndarray:
        """
        Solve the min-cost flow problem using delta-scaling.
        
        Args:
            n: Number of requesters
            m: Number of taxis
            W: Edge weights matrix (negative of costs)
            c: Cost coefficients
            d: Constant parameter
            acceptance_func: Type of acceptance function ('PL' or 'Sigmoid')
            trip_amounts: Trip valuations
            
        Returns:
            Array of prices for each requester
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes: requesters (0 to n-1), taxis (n to n+m-1), source (n+m), sink (n+m+1)
        G.add_nodes_from(range(n + m + 2))
        
        # Initialize cost matrix
        cost_matrix = np.ones((n+m+2, n+m+2)) * np.inf
        
        # Set edge costs between requesters and taxis
        for i in range(n):
            for j in range(m):
                cost_matrix[i, n+j] = -W[i, j]
                cost_matrix[n+j, i] = W[i, j]
                G.add_edge(i, n+j)
                G.add_edge(n+j, i)
        
        # Add edges from source to requesters and from taxis to sink
        for i in range(n):
            G.add_edge(n+m, i)
            G.add_edge(i, n+m)
        
        for j in range(m):
            G.add_edge(n+j, n+m+1)
            G.add_edge(n+m+1, n+j)
        
        # Add edge from source to sink
        G.add_edge(n+m, n+m+1)
        G.add_edge(n+m+1, n+m)
        
        # Initialize capacity matrix
        cap_matrix = np.zeros((n+m+2, n+m+2))
        
        # Set capacities
        for i in range(n):
            for j in range(m):
                cap_matrix[i, n+j] = np.inf
            cap_matrix[n+m, i] = 1
        
        for j in range(m):
            cap_matrix[n+j, n+m+1] = 1
        
        cap_matrix[n+m, n+m+1] = n
        
        # Initialize flow and excess
        flow = np.zeros((n+m+2, n+m+2))
        excess = np.zeros(n+m+2)
        excess[n+m] = n
        excess[n+m+1] = -n
        
        # Initialize delta
        delta = n * self.initial_delta_multiplier
        
        # Initialize potentials
        potential = np.zeros(n+m+2)
        
        # Update cost matrix for source-requester edges
        for i in range(n):
            if acceptance_func == 'PL':
                val = (1/c[i] * (delta**2) - (d/c[i]) * delta) / delta
            else:  # Sigmoid - simplified initial cost
                val = np.inf if delta == 0 else 0
                flow[n+m, i] = 0.5  # Initial flow for sigmoid
                excess[i] += 0.5
                excess[n+m] -= 0.5
                cap_matrix[n+m, i] = 0.5
                cap_matrix[i, n+m] = 0.5
            cost_matrix[n+m, i] = val
        
        # Delta-scaling iterations
        while delta > self.delta_threshold:
            # Delta-scaling phase
            self._delta_scaling_phase(
                G, n, m, delta, cost_matrix, cap_matrix, 
                flow, excess, potential, c, d, acceptance_func, trip_amounts
            )
            
            # Shortest path phase
            while self._has_excess(excess, delta):
                self._shortest_path_phase(
                    G, n, m, delta, cost_matrix, cap_matrix,
                    flow, excess, potential, c, d, acceptance_func, trip_amounts
                )
            
            # Update delta
            delta = delta / 2
            
            # Update cost matrix for new delta
            for i in range(n):
                self._update_cost(
                    i, flow[n+m, i], delta, c[i], d, 
                    cost_matrix, n, m, acceptance_func, trip_amounts[i]
                )
        
        # Compute prices from flow
        prices = self._compute_prices_from_flow(
            flow, n, m, c, d, acceptance_func, trip_amounts
        )
        
        return prices
    
    def _delta_scaling_phase(
        self, G, n, m, delta, cost_matrix, cap_matrix, 
        flow, excess, potential, c, d, acceptance_func, trip_amounts
    ):
        """Execute the delta-scaling phase."""
        # Check for negative reduced cost edges
        for i in range(n):
            for j in range(m):
                # Forward edge
                reduced_cost = cost_matrix[i, n+j] - potential[i] + potential[n+j]
                if reduced_cost < -self.epsilon and cap_matrix[i, n+j] >= delta:
                    amount = min(delta, cap_matrix[i, n+j])
                    flow[i, n+j] += amount
                    excess[i] -= amount
                    excess[n+j] += amount
                    cap_matrix[i, n+j] -= amount
                    cap_matrix[n+j, i] += amount
                
                # Backward edge
                reduced_cost = cost_matrix[n+j, i] - potential[n+j] + potential[i]
                if reduced_cost < -self.epsilon and cap_matrix[n+j, i] >= delta:
                    amount = min(delta, cap_matrix[n+j, i])
                    flow[i, n+j] -= amount
                    excess[i] += amount
                    excess[n+j] -= amount
                    cap_matrix[i, n+j] += amount
                    cap_matrix[n+j, i] -= amount
        
        # Check source-requester edges
        for i in range(n):
            # Forward edge
            reduced_cost = cost_matrix[n+m, i] - potential[n+m] + potential[i]
            if reduced_cost < -self.epsilon and cap_matrix[n+m, i] >= delta:
                amount = min(delta, cap_matrix[n+m, i])
                flow[n+m, i] += amount
                excess[n+m] -= amount
                excess[i] += amount
                cap_matrix[n+m, i] -= amount
                cap_matrix[i, n+m] += amount
                self._update_cost(
                    i, flow[n+m, i], delta, c[i], d,
                    cost_matrix, n, m, acceptance_func, trip_amounts[i]
                )
        
        # Check taxi-sink edges
        for j in range(m):
            reduced_cost = -potential[n+j] + potential[n+m+1]
            if reduced_cost < -self.epsilon and cap_matrix[n+j, n+m+1] >= delta:
                amount = min(delta, cap_matrix[n+j, n+m+1])
                flow[n+j, n+m+1] += amount
                excess[n+j] -= amount
                excess[n+m+1] += amount
                cap_matrix[n+j, n+m+1] -= amount
                cap_matrix[n+m+1, n+j] += amount
    
    def _shortest_path_phase(
        self, G, n, m, delta, cost_matrix, cap_matrix,
        flow, excess, potential, c, d, acceptance_func, trip_amounts
    ):
        """Execute the shortest path phase using Dijkstra's algorithm."""
        # Find node with excess
        start_node = None
        for i in range(n+m+2):
            if excess[i] >= delta:
                start_node = i
                break
        
        if start_node is None:
            return
        
        # Dijkstra's algorithm
        node_num = n + m + 2
        distance = [math.inf] * node_num
        previous = [-1] * node_num
        distance[start_node] = 0
        
        unvisited = list(range(node_num))
        visited = []
        
        while unvisited:
            # Find minimum distance node
            min_dist = math.inf
            min_node = -1
            for node in unvisited:
                if distance[node] < min_dist:
                    min_dist = distance[node]
                    min_node = node
            
            if min_node == -1:
                break
                
            unvisited.remove(min_node)
            visited.append(min_node)
            
            # Check if we found a deficit node
            if excess[min_node] <= -delta:
                end_node = min_node
                break
            
            # Update distances to neighbors
            for neighbor in G.successors(min_node):
                if cap_matrix[min_node, neighbor] >= delta:
                    reduced_cost = (cost_matrix[min_node, neighbor] - 
                                  potential[min_node] + potential[neighbor])
                    new_dist = distance[min_node] + reduced_cost
                    
                    if new_dist < distance[neighbor]:
                        distance[neighbor] = new_dist
                        previous[neighbor] = min_node
        
        # Update potentials
        for i in range(node_num):
            if i in visited:
                potential[i] -= distance[i]
            else:
                potential[i] -= distance[end_node] if 'end_node' in locals() else 0
        
        # Augment flow along path
        if 'end_node' in locals():
            # Trace back path
            path = []
            current = end_node
            while current != start_node:
                prev = previous[current]
                path.append((prev, current))
                current = prev
            
            # Augment flow
            for (u, v) in reversed(path):
                flow[u, v] += delta
                cap_matrix[u, v] -= delta
                cap_matrix[v, u] += delta
                
                # Update cost if it's a source-requester edge
                if u == n+m and v < n:
                    self._update_cost(
                        v, flow[n+m, v], delta, c[v], d,
                        cost_matrix, n, m, acceptance_func, trip_amounts[v]
                    )
            
            # Update excess
            excess[start_node] -= delta
            excess[end_node] += delta
    
    def _has_excess(self, excess: np.ndarray, delta: float) -> bool:
        """Check if there are nodes with excess and deficit."""
        has_positive = any(excess >= delta)
        has_negative = any(excess <= -delta)
        return has_positive and has_negative
    
    def _update_cost(
        self, i: int, flow_i: float, delta: float, 
        c_i: float, d: float, cost_matrix: np.ndarray,
        n: int, m: int, acceptance_func: str, trip_amount: float
    ):
        """Update the cost for source-requester edges."""
        if acceptance_func == 'PL':
            # Piecewise linear cost
            if flow_i + delta > 0:
                val_plus = ((1/c_i) * ((flow_i + delta)**2) - 
                           (d/c_i) * (flow_i + delta) -
                           (1/c_i) * (flow_i**2) + (d/c_i) * flow_i) / delta
            else:
                val_plus = 0
                
            if flow_i - delta > 0:
                val_minus = ((1/c_i) * ((flow_i - delta)**2) - 
                            (d/c_i) * (flow_i - delta) -
                            (1/c_i) * (flow_i**2) + (d/c_i) * flow_i) / delta
            else:
                val_minus = 0
                
            cost_matrix[n+m, i] = val_plus
            cost_matrix[i, n+m] = val_minus
        else:
            # Sigmoid cost - using simplified version
            gamma = self.config.get('sigmoid_gamma', 0.276)
            beta = d  # beta parameter
            
            if flow_i > 0 and flow_i < 1:
                # Compute derivative of cost function
                val_plus = self._sigmoid_cost_derivative(
                    flow_i, delta, gamma, beta, trip_amount, True
                )
                val_minus = self._sigmoid_cost_derivative(
                    flow_i, delta, gamma, beta, trip_amount, False
                )
                cost_matrix[n+m, i] = val_plus
                cost_matrix[i, n+m] = val_minus
    
    def _sigmoid_cost_derivative(
        self, flow: float, delta: float, gamma: float, 
        beta: float, trip_amount: float, is_forward: bool
    ) -> float:
        """Compute the derivative of sigmoid cost function."""
        if is_forward:
            new_flow = flow + delta
        else:
            new_flow = flow - delta
            
        if new_flow <= 0 or new_flow >= 1:
            return np.inf
            
        # Cost function derivative for sigmoid
        if flow > 0:
            old_cost = (gamma * trip_amount * np.log(flow/(1-flow)) * flow - 
                       beta * trip_amount * flow)
        else:
            old_cost = 0
            
        new_cost = (gamma * trip_amount * np.log(new_flow/(1-new_flow)) * new_flow - 
                   beta * trip_amount * new_flow)
        
        return (new_cost - old_cost) / delta
    
    def _compute_prices_from_flow(
        self, flow: np.ndarray, n: int, m: int,
        c: np.ndarray, d: float, acceptance_func: str, trip_amounts: np.ndarray
    ) -> np.ndarray:
        """Compute prices from the optimal flow."""
        prices = np.zeros(n)
        
        for i in range(n):
            flow_i = flow[n+m, i]
            
            if acceptance_func == 'PL':
                # For PL: price = -(1/c)*flow + d/c
                prices[i] = -(1/c[i]) * flow_i + d/c[i]
            else:
                # For Sigmoid: price = -gamma*trip_amount*log(flow/(1-flow)) + beta*trip_amount
                gamma = self.config.get('sigmoid_gamma', 0.276)
                beta = d
                
                if flow_i > 0 and flow_i < 1:
                    prices[i] = (-gamma * trip_amounts[i] * 
                                np.log(flow_i/(1-flow_i)) + 
                                beta * trip_amounts[i])
                else:
                    # Default price if flow is at boundary
                    prices[i] = beta * trip_amounts[i]
        
        # Ensure prices are non-negative
        prices = np.maximum(prices, 0)
        
        return prices 