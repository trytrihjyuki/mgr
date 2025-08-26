"""MAPS pricing method implementation (Baseline from Tong et al.).

This implements the MAPS (Matching And Pricing in Shared economy) algorithm
used as a baseline in Hikima et al.'s experiments.
"""

import numpy as np
from typing import Dict, Any, Set, List, Tuple
from .base import BasePricingMethod


class MAPSMethod(BasePricingMethod):
    """
    MAPS (Matching And Pricing in Shared economy) pricing method.
    
    This is a greedy algorithm that iteratively matches requesters to taxis
    while optimizing prices for each area/region.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MAPS pricing method."""
        super().__init__(config)
        
        # MAPS-specific parameters
        self.s0_rate = config.get('maps_s0_rate', 1.5)
        self.price_delta = config.get('maps_price_delta', 0.05)
        self.matching_radius = config.get('maps_matching_radius', 2.0)  # km
        
    def get_method_name(self) -> str:
        """Get method name."""
        return "MAPS"
    
    def compute_prices(self, scenario_data: Dict[str, Any]) -> np.ndarray:
        """
        Compute prices using the MAPS algorithm.
        
        Args:
            scenario_data: Dictionary with scenario data
            
        Returns:
            Array of prices for each requester
        """
        n_requesters = scenario_data['num_requesters']
        n_taxis = scenario_data['num_taxis']
        edge_weights = scenario_data['edge_weights']
        trip_amounts = scenario_data['trip_amounts']
        trip_distances = scenario_data.get('trip_distances', np.ones(n_requesters))
        
        # Handle edge cases
        if n_requesters == 0:
            return np.array([])
        if n_taxis == 0:
            return np.zeros(n_requesters)
        
        # Get location IDs if available
        location_ids = scenario_data.get('location_ids', np.arange(n_requesters))
        
        # Compute distance matrix (convert from edge weights)
        distance_matrix = -edge_weights / self.alpha * self.s_taxi
        
        # Run MAPS algorithm
        prices = self._maps_algorithm(
            n_requesters, n_taxis, distance_matrix,
            trip_amounts, trip_distances, location_ids
        )
        
        return prices
    
    def _maps_algorithm(
        self,
        n_requesters: int,
        n_taxis: int,
        distance_matrix: np.ndarray,
        trip_amounts: np.ndarray,
        trip_distances: np.ndarray,
        location_ids: np.ndarray
    ) -> np.ndarray:
        """
        Execute the MAPS algorithm.
        
        Args:
            n_requesters: Number of requesters
            n_taxis: Number of taxis
            distance_matrix: Distance matrix between requesters and taxis
            trip_amounts: Trip valuations
            trip_distances: Trip distances
            location_ids: Location IDs for grouping requesters
            
        Returns:
            Array of prices for each requester
        """
        # Get unique areas
        unique_areas = np.unique(location_ids)
        n_areas = len(unique_areas)
        
        # Initialize prices (per unit distance) for each area
        p_max = np.max(trip_amounts / trip_distances) * self.s0_rate
        p_min = np.min(trip_amounts / trip_distances)
        
        # Current prices for each area (start with max price)
        area_prices = {area: p_max for area in unique_areas}
        
        # Number of requesters matched in each area
        matched_count = {area: 0 for area in unique_areas}
        
        # Maximum number of requesters in each area
        max_requesters = {
            area: np.sum(location_ids == area) 
            for area in unique_areas
        }
        
        # Sum of distances for requesters in each area
        distance_sum = {
            area: np.sum(trip_distances[location_ids == area])
            for area in unique_areas
        }
        
        # Build price grid for each area
        n_price_levels = int(np.log(p_max/p_min) / np.log(1 + self.price_delta)) + 1
        price_grids = {}
        for area in unique_areas:
            prices = [p_max * ((1/(1+self.price_delta))**i) for i in range(n_price_levels)]
            price_grids[area] = np.array(prices)
        
        # Compute average acceptance rates for each price level
        acceptance_rates = self._compute_acceptance_rates(
            unique_areas, price_grids, trip_amounts, 
            trip_distances, location_ids
        )
        
        # Build bipartite matching edges (within matching radius)
        edges = self._build_matching_edges(
            n_requesters, n_taxis, distance_matrix
        )
        
        # Track matched requesters and taxis
        matched_requesters = set()
        matched_taxis = set()
        
        # Greedy matching iterations
        while True:
            # Compute delta values for each area
            deltas = self._compute_deltas(
                unique_areas, area_prices, matched_count,
                max_requesters, distance_sum, price_grids,
                acceptance_rates
            )
            
            # Find area with maximum delta
            if not deltas or max(deltas.values()) <= 0:
                break
                
            max_area = max(deltas, key=deltas.get)
            
            # Try to find an augmenting path for this area
            aug_path_found = False
            for i in range(n_requesters):
                if (i not in matched_requesters and 
                    location_ids[i] == max_area):
                    # Try to match this requester
                    path = self._find_augmenting_path(
                        i, edges, matched_taxis, n_requesters
                    )
                    if path is not None:
                        # Update matching
                        matched_requesters.add(i)
                        matched_taxis.add(path[-1])
                        matched_count[max_area] += 1
                        aug_path_found = True
                        break
            
            if not aug_path_found:
                # No more matching possible for this area
                deltas[max_area] = -1
            else:
                # Update price for the area
                new_price_idx = self._find_optimal_price_index(
                    max_area, matched_count[max_area],
                    max_requesters[max_area], distance_sum[max_area],
                    price_grids[max_area], acceptance_rates[max_area]
                )
                if new_price_idx >= 0:
                    area_prices[max_area] = price_grids[max_area][new_price_idx]
        
        # Convert area prices to individual requester prices
        prices = np.zeros(n_requesters)
        for i in range(n_requesters):
            area = location_ids[i]
            # Price = price_per_km * distance
            prices[i] = area_prices[area] * trip_distances[i]
        
        return prices
    
    def _compute_acceptance_rates(
        self,
        areas: np.ndarray,
        price_grids: Dict,
        trip_amounts: np.ndarray,
        trip_distances: np.ndarray,
        location_ids: np.ndarray
    ) -> Dict:
        """Compute acceptance rates for each area and price level."""
        acceptance_rates = {}
        # Use PL acceptance function for MAPS optimization
        # The actual acceptance will be computed in the base class for both functions
        
        for area in areas:
            area_mask = location_ids == area
            area_amounts = trip_amounts[area_mask]
            area_distances = trip_distances[area_mask]
            n_area = len(area_amounts)
            
            if n_area == 0:
                acceptance_rates[area] = np.zeros(len(price_grids[area]))
                continue
            
            rates = []
            for price in price_grids[area]:
                total_rate = 0
                for amount, dist in zip(area_amounts, area_distances):
                    total_price = price * dist
                    
                    # Use PL acceptance for optimization
                    # Piecewise linear acceptance
                    rate = max(0, min(1, -2.0/amount * total_price + 3.0))
                    
                    total_rate += rate
                
                rates.append(total_rate / n_area)
            
            acceptance_rates[area] = np.array(rates)
        
        return acceptance_rates
    
    def _build_matching_edges(
        self,
        n_requesters: int,
        n_taxis: int,
        distance_matrix: np.ndarray
    ) -> List[Set[int]]:
        """Build bipartite matching edges within matching radius."""
        edges = [set() for _ in range(n_requesters)]
        
        for i in range(n_requesters):
            for j in range(n_taxis):
                if distance_matrix[i, j] <= self.matching_radius:
                    edges[i].add(j)
        
        return edges
    
    def _compute_deltas(
        self,
        areas: np.ndarray,
        current_prices: Dict,
        matched_count: Dict,
        max_requesters: Dict,
        distance_sum: Dict,
        price_grids: Dict,
        acceptance_rates: Dict
    ) -> Dict:
        """Compute delta values for each area."""
        deltas = {}
        
        for area in areas:
            if matched_count[area] >= max_requesters[area]:
                deltas[area] = -1
                continue
            
            # Find current price index
            current_price = current_prices[area]
            current_idx = np.argmin(np.abs(price_grids[area] - current_price))
            
            # Find optimal price for matched_count + 1
            opt_idx = self._find_optimal_price_index(
                area, matched_count[area] + 1,
                max_requesters[area], distance_sum[area],
                price_grids[area], acceptance_rates[area]
            )
            
            if opt_idx < 0:
                deltas[area] = -1
                continue
            
            # Compute delta
            current_value = ((current_price - self.alpha/self.s_taxi) * 
                           acceptance_rates[area][current_idx])
            new_value = ((price_grids[area][opt_idx] - self.alpha/self.s_taxi) * 
                        acceptance_rates[area][opt_idx])
            deltas[area] = new_value - current_value
        
        return deltas
    
    def _find_optimal_price_index(
        self,
        area: Any,
        n_matched: int,
        max_n: int,
        dist_sum: float,
        prices: np.ndarray,
        rates: np.ndarray
    ) -> int:
        """Find optimal price index for given number of matches."""
        if n_matched > max_n:
            return -1
        
        best_value = -np.inf
        best_idx = -1
        
        # Compute total distance for n_matched requesters (simplified)
        avg_dist = dist_sum / max_n if max_n > 0 else 1
        total_dist = avg_dist * n_matched
        
        for idx, (price, rate) in enumerate(zip(prices, rates)):
            # Compute value: min(capacity_value, demand_value)
            capacity_value = dist_sum * (price - self.alpha/self.s_taxi) * rate
            demand_value = total_dist * (price - self.alpha/self.s_taxi)
            value = min(capacity_value, demand_value)
            
            if value > best_value:
                best_value = value
                best_idx = idx
        
        return best_idx
    
    def _find_augmenting_path(
        self,
        requester: int,
        edges: List[Set[int]],
        matched_taxis: Set[int],
        n_requesters: int
    ) -> List[int]:
        """Find an augmenting path using DFS."""
        # Simple DFS to find unmatched taxi
        visited = set()
        
        def dfs(req_idx):
            for taxi_idx in edges[req_idx]:
                if taxi_idx in visited:
                    continue
                visited.add(taxi_idx)
                
                if taxi_idx not in matched_taxis:
                    return [req_idx, taxi_idx]
                    
                # For simplicity, we only look for direct unmatched taxis
                # In full implementation, this would search for alternating paths
            
            return None
        
        return dfs(requester) 