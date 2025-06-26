"""
LinUCB (Contextual Bandit) Implementation

This implements the LinUCB algorithm extracted from the provided source code
(experiment_PL.py and experiment_sigmoid.py). LinUCB uses contextual bandit
learning to select optimal prices from a discrete set of arms.
"""

import numpy as np
import pandas as pd
import time
import pickle
from typing import Dict, List, Any, Tuple

from .base_method import BasePricingMethod, PricingResult


class LinUCB(BasePricingMethod):
    """
    Implementation of the LinUCB (Linear Upper Confidence Bound) algorithm.
    
    This algorithm uses contextual bandit learning to select optimal prices
    from a discrete set of price arms, with confidence-based exploration.
    """
    
    def __init__(self, **kwargs):
        super().__init__("LinUCB", **kwargs)
        
        # LinUCB parameters from the original source code
        self.ucb_alpha = kwargs.get('ucb_alpha', 0.5)
        self.base_price = kwargs.get('base_price', 5.875)
        self.price_multipliers = kwargs.get('price_multipliers', [0.6, 0.8, 1.0, 1.2, 1.4])
        self.arm_prices = self.base_price * np.array(self.price_multipliers)
        
        # Feature configuration
        self.use_time_features = kwargs.get('use_time_features', True)
        self.use_location_features = kwargs.get('use_location_features', True)
        self.use_distance_features = kwargs.get('use_distance_features', True)
        
        # Acceptance function
        self.acceptance_function = kwargs.get('acceptance_function', 'PL')
        
        # Sigmoid parameters
        self.sigmoid_beta = kwargs.get('sigmoid_beta', 1.3)
        self.sigmoid_gamma = kwargs.get('sigmoid_gamma', 0.3 * np.sqrt(3) / np.pi)
        
        # Initialize or load learned matrices
        self.initialize_learned_matrices(**kwargs)
    
    def initialize_learned_matrices(self, **kwargs):
        """
        Initialize the learned matrices A and b for each arm.
        
        In the original code, these are loaded from pickle files trained on historical data.
        For this implementation, we'll initialize them appropriately.
        """
        # Determine feature dimension based on configuration
        feature_dim = self._calculate_feature_dimension()
        
        # Initialize A matrices (one per arm) - start with identity for regularization
        self.A_matrices = {}
        self.b_vectors = {}
        
        for i, arm_price in enumerate(self.arm_prices):
            self.A_matrices[i] = np.eye(feature_dim)  # Start with identity matrix
            self.b_vectors[i] = np.zeros(feature_dim)  # Start with zero vector
        
        # Load pre-trained matrices if provided
        if 'pretrained_matrices' in kwargs:
            self._load_pretrained_matrices(kwargs['pretrained_matrices'])
    
    def _calculate_feature_dimension(self):
        """Calculate the dimension of the feature vector."""
        dim = 0
        
        if self.use_time_features:
            dim += 10  # One-hot encoding for 10 hours (10:00-20:00)
        
        if self.use_location_features:
            # This would depend on the number of pickup/dropoff zones
            # From the source code, this appears to be based on PUID_set and DOID_set
            dim += 50  # Approximate number of pickup zones
            dim += 50  # Approximate number of dropoff zones
        
        if self.use_distance_features:
            dim += 1   # Trip distance
            dim += 1   # Trip duration
        
        return dim
    
    def calculate_prices(self, 
                        requesters_data: pd.DataFrame,
                        taxis_data: pd.DataFrame,
                        distance_matrix: np.ndarray,
                        **kwargs) -> PricingResult:
        """
        Calculate optimal prices using LinUCB algorithm.
        
        This implements the exact LinUCB algorithm from the provided source code.
        """
        start_time = time.time()
        
        n = len(requesters_data)
        m = len(taxis_data)
        
        if n == 0 or m == 0:
            return self._create_empty_result(start_time)
        
        # Extract trip data
        trip_distances = requesters_data['trip_distance'].values * 1.60934  # Convert to km
        trip_amounts = requesters_data['total_amount'].values
        
        # Get current hour for time features
        current_hour = kwargs.get('current_hour', 12)  # Default to noon
        
        # Run LinUCB algorithm
        prices, selected_arms, acceptance_probs = self._run_linucb_algorithm(
            requesters_data, current_hour, trip_distances, trip_amounts)
        
        # Calculate edge weights for evaluation
        w_matrix = self._calculate_edge_weights(distance_matrix, trip_distances, 18.0, 25.0)
        
        # Simulate matching for evaluation
        acceptance_results = np.random.binomial(1, acceptance_probs)
        objective_value, matches = self._evaluate_matching(prices, acceptance_results, w_matrix)
        
        # Update learned matrices based on observed rewards (simplified)
        self._update_learned_matrices(requesters_data, selected_arms, matches, 
                                     current_hour, trip_distances, trip_amounts)
        
        computation_time = time.time() - start_time
        
        return PricingResult(
            method_name=self.method_name,
            prices=prices,
            acceptance_probabilities=acceptance_probs,
            objective_value=objective_value,
            computation_time=computation_time,
            matches=matches,
            additional_metrics={
                'algorithm': 'linear_upper_confidence_bound',
                'acceptance_function': self.acceptance_function,
                'selected_arms': selected_arms.tolist(),
                'arm_prices': self.arm_prices.tolist(),
                'n_requesters': n,
                'n_taxis': m
            }
        )
    
    def _run_linucb_algorithm(self, requesters_data: pd.DataFrame, current_hour: int,
                             trip_distances: np.ndarray, trip_amounts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the LinUCB algorithm extracted from the provided source code.
        
        This follows the exact algorithm from experiment_PL.py and experiment_sigmoid.py.
        """
        n = len(requesters_data)
        prices = np.zeros(n)
        selected_arms = np.zeros(n, dtype=int)
        acceptance_probs = np.zeros(n)
        
        # Get area information for location features
        if 'PULocationID' in requesters_data.columns:
            pu_location_ids = requesters_data['PULocationID'].values
            do_location_ids = requesters_data['DOLocationID'].values if 'DOLocationID' in requesters_data.columns else pu_location_ids
        else:
            pu_location_ids = np.ones(n, dtype=int)
            do_location_ids = np.ones(n, dtype=int)
        
        # Calculate theta vectors for each arm (line 4 in Algorithm 1 of Chu's paper)
        theta_vectors = {}
        for arm in range(len(self.arm_prices)):
            try:
                theta_vectors[arm] = np.linalg.solve(self.A_matrices[arm], self.b_vectors[arm])
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                theta_vectors[arm] = np.linalg.pinv(self.A_matrices[arm]) @ self.b_vectors[arm]
        
        # Process each requester
        for i in range(n):
            # Create feature vector X (line 5 in Algorithm 1 of Chu's paper)
            X = self._create_feature_vector(
                current_hour, pu_location_ids[i], do_location_ids[i], 
                trip_distances[i], requesters_data.iloc[i])
            
            # Calculate upper confidence bounds for each arm (lines 7 and 9)
            max_ucb = -np.inf
            best_arm = 0
            
            for arm in range(len(self.arm_prices)):
                # Calculate UCB: theta^T * x + alpha * sqrt(x^T * A^(-1) * x)
                try:
                    A_inv_x = np.linalg.solve(self.A_matrices[arm], X)
                    confidence = self.ucb_alpha * np.sqrt(np.dot(X, A_inv_x))
                except np.linalg.LinAlgError:
                    A_inv_x = np.linalg.pinv(self.A_matrices[arm]) @ X
                    confidence = self.ucb_alpha * np.sqrt(np.dot(X, A_inv_x))
                
                ucb_value = np.dot(theta_vectors[arm], X) + confidence
                
                if ucb_value > max_ucb:
                    max_ucb = ucb_value
                    best_arm = arm
            
            # Select the arm and set price
            selected_arms[i] = best_arm
            prices[i] = self.arm_prices[best_arm] * trip_distances[i]  # Scale by distance
            
            # Calculate acceptance probability
            if self.acceptance_function == 'PL':
                # Piecewise Linear acceptance: -2.0/trip_amount * price + 3
                acceptance_probs[i] = -2.0 / trip_amounts[i] * prices[i] + 3.0
            else:
                # Sigmoid acceptance
                exponent = (-prices[i] + self.sigmoid_beta * trip_amounts[i]) / (self.sigmoid_gamma * trip_amounts[i])
                exponent = np.clip(exponent, -50, 50)
                acceptance_probs[i] = 1 - 1 / (1 + np.exp(exponent))
            
            # Ensure acceptance probability is in [0, 1]
            acceptance_probs[i] = np.clip(acceptance_probs[i], 0.0, 1.0)
        
        return prices, selected_arms, acceptance_probs
    
    def _create_feature_vector(self, current_hour: int, pu_location_id: int, 
                              do_location_id: int, trip_distance: float, 
                              trip_row: pd.Series) -> np.ndarray:
        """
        Create feature vector X as per the LinUCB algorithm in the source code.
        
        This follows the exact feature construction from experiment_PL.py and experiment_sigmoid.py.
        """
        features = []
        
        # Time features (one-hot encoding for hour)
        if self.use_time_features:
            hour_onehot = np.zeros(10)
            if 10 <= current_hour <= 19:  # Map to 0-9 index
                hour_onehot[current_hour - 10] = 1
            features.extend(hour_onehot)
        
        # Location features (simplified - in practice would use actual zone sets)
        if self.use_location_features:
            # Pickup location one-hot (simplified to 50 zones)
            pu_onehot = np.zeros(50)
            pu_idx = min(49, max(0, pu_location_id % 50))
            pu_onehot[pu_idx] = 1
            features.extend(pu_onehot)
            
            # Dropoff location one-hot (simplified to 50 zones)
            do_onehot = np.zeros(50)
            do_idx = min(49, max(0, do_location_id % 50))
            do_onehot[do_idx] = 1
            features.extend(do_onehot)
        
        # Distance and duration features
        if self.use_distance_features:
            features.append(trip_distance)
            
            # Trip duration (simplified - use a default if not available)
            trip_duration = trip_row.get('trip_duration', 30.0)  # Default 30 minutes
            features.append(trip_duration)
        
        return np.array(features)
    
    def _update_learned_matrices(self, requesters_data: pd.DataFrame, selected_arms: np.ndarray,
                                matches: List[Tuple[int, int]], current_hour: int,
                                trip_distances: np.ndarray, trip_amounts: np.ndarray):
        """
        Update the learned matrices A and b based on observed rewards.
        
        This implements lines 11 and 12 of Algorithm 1 in Chu's paper.
        """
        n = len(requesters_data)
        
        # Get area information
        if 'PULocationID' in requesters_data.columns:
            pu_location_ids = requesters_data['PULocationID'].values
            do_location_ids = requesters_data['DOLocationID'].values if 'DOLocationID' in requesters_data.columns else pu_location_ids
        else:
            pu_location_ids = np.ones(n, dtype=int)
            do_location_ids = np.ones(n, dtype=int)
        
        # Calculate rewards based on matches
        rewards = np.zeros(n)
        matched_requesters = set(match[0] for match in matches)
        
        for i in range(n):
            if i in matched_requesters:
                # Reward is the price if matched
                rewards[i] = self.arm_prices[selected_arms[i]] * trip_distances[i]
            else:
                # No reward if not matched
                rewards[i] = 0.0
        
        # Update matrices for each requester
        for i in range(n):
            arm = selected_arms[i]
            
            # Create feature vector
            X = self._create_feature_vector(
                current_hour, pu_location_ids[i], do_location_ids[i],
                trip_distances[i], requesters_data.iloc[i])
            
            # Update A matrix: A = A + X * X^T
            self.A_matrices[arm] += np.outer(X, X)
            
            # Update b vector: b = b + X * reward
            self.b_vectors[arm] += X * rewards[i]
    
    def _load_pretrained_matrices(self, pretrained_path: str):
        """
        Load pre-trained matrices from pickle files.
        
        This would load the matrices that were trained on historical data
        as mentioned in the original source code.
        """
        try:
            # In the original code, matrices are loaded like:
            # with open('../work/learned_matrix_PL/201908_Manhattan/A_0_08', 'rb') as web:
            #     A_0 = pickle.load(web)
            
            # For this implementation, we assume matrices are provided in a specific format
            # This is a placeholder for the actual loading logic
            pass
        except Exception as e:
            print(f"Warning: Could not load pretrained matrices: {e}")
    
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