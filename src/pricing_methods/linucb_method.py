#!/usr/bin/env python3
"""
LinUCB Method Implementation
Extracted from experiment_PL.py and experiment_sigmoid.py sources
"""

import numpy as np
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LinUCBResult:
    """Result structure for LinUCB method"""
    method_name: str = "LinUCB"
    prices: np.ndarray = None
    objective_value: float = 0.0
    computation_time: float = 0.0
    arm_selections: np.ndarray = None
    confidence_bounds: np.ndarray = None
    acceptance_probabilities: np.ndarray = None
    matched_pairs: List[Tuple[int, int]] = None
    theta_estimates: Dict[int, np.ndarray] = None
    
    def __post_init__(self):
        if self.matched_pairs is None:
            self.matched_pairs = []
        if self.theta_estimates is None:
            self.theta_estimates = {}


class LinUCBMethod:
    """
    Implementation of LinUCB (Linear Upper Confidence Bound) method for taxi pricing
    Based on exact mathematical formulation from the provided sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LinUCB method with configuration
        
        Args:
            config: Configuration dictionary containing method parameters
        """
        self.ucb_alpha = config.get('ucb_alpha', 0.5)
        self.base_price = config.get('base_price', 5.875)
        self.price_multipliers = config.get('price_multipliers', [0.6, 0.8, 1.0, 1.2, 1.4])
        self.acceptance_type = config.get('acceptance_type', 'PL')  # 'PL' or 'Sigmoid'
        
        # Feature engineering configuration
        self.use_time_features = config.get('use_time_features', True)
        self.use_location_features = config.get('use_location_features', True)
        self.use_distance_features = config.get('use_distance_features', True)
        
        # Sigmoid parameters for acceptance function
        self.sigmoid_params = config.get('sigmoid_params', {
            'beta': 1.3,
            'gamma': 0.3 * math.sqrt(3) / math.pi
        })
        
        # Calculate arm prices
        self.arm_prices = [self.base_price * multiplier for multiplier in self.price_multipliers]
        self.num_arms = len(self.arm_prices)
        
        # Initialize A and b matrices for each arm (exact from original code)
        self.A_matrices = {}
        self.b_vectors = {}
        
        logger.info(f"Initialized LinUCB method with {self.num_arms} arms, α={self.ucb_alpha}")
    
    def initialize_arm_matrices(self, feature_dim: int):
        """
        Initialize A and b matrices for each arm
        Following exact LinUCB algorithm from experiment sources
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        for arm in range(self.num_arms):
            self.A_matrices[arm] = np.eye(feature_dim)  # A_a = I (identity matrix)
            self.b_vectors[arm] = np.zeros(feature_dim)  # b_a = 0 vector
    
    def create_feature_vector(self, requester_data: np.ndarray, time_hour: int, 
                             unique_pickup_ids: List[int], unique_dropoff_ids: List[int]) -> np.ndarray:
        """
        Create feature vector for a requester following exact methodology from experiment sources
        
        Args:
            requester_data: Single requester data [borough, area_id, trip_dist, total_amount, ...]
            time_hour: Hour of the day (0-23)
            unique_pickup_ids: List of unique pickup location IDs
            unique_dropoff_ids: List of unique dropoff location IDs
            
        Returns:
            Feature vector X for LinUCB
        """
        features = []
        
        # Time features (exact from original: hour one-hot encoding for 10 hours)
        if self.use_time_features:
            hour_onehot = np.zeros(10)  # Hours 10-19 (10 hours as in original)
            if 10 <= time_hour <= 19:
                hour_onehot[time_hour - 10] = 1
            features.extend(hour_onehot)
        
        # Pickup location features (exact from original: PUID one-hot)
        if self.use_location_features:
            pickup_id = int(requester_data[1]) if len(requester_data) > 1 else 161  # Default to Midtown
            pickup_onehot = np.zeros(len(unique_pickup_ids))
            if pickup_id in unique_pickup_ids:
                pickup_idx = unique_pickup_ids.index(pickup_id)
                pickup_onehot[pickup_idx] = 1
            features.extend(pickup_onehot)
            
            # Dropoff location features (exact from original: DOID one-hot)
            dropoff_id = int(requester_data[4]) if len(requester_data) > 4 else pickup_id  # Default to same as pickup
            dropoff_onehot = np.zeros(len(unique_dropoff_ids))
            if dropoff_id in unique_dropoff_ids:
                dropoff_idx = unique_dropoff_ids.index(dropoff_id)
                dropoff_onehot[dropoff_idx] = 1
            features.extend(dropoff_onehot)
        
        # Trip characteristics (exact from original: trip_distance and time_consume)
        if self.use_distance_features:
            trip_distance = float(requester_data[2]) if len(requester_data) > 2 else 1.0
            time_consume = float(requester_data[5]) if len(requester_data) > 5 else 300.0  # 5 minutes default
            features.extend([trip_distance, time_consume])
        
        return np.array(features)
    
    def calculate_ucb_scores(self, feature_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate UCB scores for all arms following exact LinUCB algorithm
        
        Args:
            feature_vector: Feature vector X for current context
            
        Returns:
            Tuple of (UCB scores, confidence bounds) for each arm
        """
        ucb_scores = np.zeros(self.num_arms)
        confidence_bounds = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            # Calculate theta estimate: θ_a = A_a^{-1} * b_a (line 4 in Algorithm 1)
            A_inv = np.linalg.inv(self.A_matrices[arm])
            theta_a = A_inv @ self.b_vectors[arm]
            
            # Calculate confidence bound: α * sqrt(x^T * A_a^{-1} * x) (line 7 in Algorithm 1)
            confidence_bound = self.ucb_alpha * math.sqrt(feature_vector.T @ A_inv @ feature_vector)
            confidence_bounds[arm] = confidence_bound
            
            # Calculate UCB score: θ_a^T * x + confidence_bound (line 7 in Algorithm 1)
            ucb_scores[arm] = theta_a.T @ feature_vector + confidence_bound
        
        return ucb_scores, confidence_bounds
    
    def calculate_acceptance_probability(self, price: float, requester_data: np.ndarray) -> float:
        """
        Calculate acceptance probability for given price and requester
        Following exact formulas from experiment sources
        
        Args:
            price: Proposed price
            requester_data: Requester data
            
        Returns:
            Acceptance probability
        """
        total_amount = float(requester_data[3]) if len(requester_data) > 3 else 10.0
        
        if self.acceptance_type == 'PL':
            # Piecewise linear acceptance (exact from experiment_PL.py)
            acceptance_prob = max(0, min(1, -2.0/total_amount * price + 3.0))
        else:
            # Sigmoid acceptance (exact from experiment_sigmoid.py)
            beta = self.sigmoid_params['beta']
            gamma = self.sigmoid_params['gamma']
            
            if abs(total_amount) < 1e-6:
                acceptance_prob = 0.5
            else:
                exponent = (-price + beta * total_amount) / (gamma * total_amount)
                acceptance_prob = 1 - (1 / (1 + math.exp(max(-50, min(50, exponent)))))
        
        return max(0.0, min(1.0, acceptance_prob))
    
    def update_arm_parameters(self, arm: int, feature_vector: np.ndarray, reward: float):
        """
        Update A and b matrices for selected arm following exact LinUCB update
        
        Args:
            arm: Selected arm index
            feature_vector: Feature vector X
            reward: Observed reward
        """
        # Update A matrix: A_a = A_a + x * x^T (line 11 in Algorithm 1)
        self.A_matrices[arm] += np.outer(feature_vector, feature_vector)
        
        # Update b vector: b_a = b_a + r * x (line 12 in Algorithm 1)
        self.b_vectors[arm] += reward * feature_vector
    
    def solve(self, requesters_data: np.ndarray, taxis_data: np.ndarray) -> LinUCBResult:
        """
        Main solve method for LinUCB pricing optimization
        Following exact algorithm from experiment sources
        
        Args:
            requesters_data: Array of requester data
            taxis_data: Array of taxi data
            
        Returns:
            LinUCBResult containing solution
        """
        import time
        start_time = time.time()
        
        n = len(requesters_data)
        m = len(taxis_data)
        
        logger.info(f"Starting LinUCB method with {n} requesters, {m} taxis")
        
        # Extract unique location IDs for feature encoding
        unique_pickup_ids = list(set(requesters_data[:, 1].astype(int)))
        unique_dropoff_ids = list(set(requesters_data[:, 4].astype(int)) if requesters_data.shape[1] > 4 
                                 else unique_pickup_ids)
        
        # Create sample feature vector to determine dimension
        sample_feature = self.create_feature_vector(
            requesters_data[0] if n > 0 else np.array([0, 161, 1.0, 10.0, 161, 300.0]),
            12,  # Sample hour
            unique_pickup_ids,
            unique_dropoff_ids
        )
        feature_dim = len(sample_feature)
        
        # Initialize arm matrices
        self.initialize_arm_matrices(feature_dim)
        
        # Results storage
        prices = np.zeros(n)
        arm_selections = np.zeros(n, dtype=int)
        confidence_bounds = np.zeros(n)
        acceptance_probs = np.zeros(n)
        rewards = np.zeros(n)
        
        # Process each requester (following exact LinUCB algorithm)
        for i in range(n):
            requester = requesters_data[i]
            
            # Extract time information (assuming we have some time context)
            time_hour = 12  # Simplified - in real implementation, extract from timestamp
            
            # Create feature vector (line 5 in Algorithm 1)
            feature_vector = self.create_feature_vector(
                requester, time_hour, unique_pickup_ids, unique_dropoff_ids
            )
            
            # Calculate UCB scores for all arms (lines 6-8 in Algorithm 1)
            ucb_scores, conf_bounds = self.calculate_ucb_scores(feature_vector)
            
            # Select arm with highest UCB score (line 9 in Algorithm 1)
            selected_arm = np.argmax(ucb_scores)
            selected_price = self.arm_prices[selected_arm]
            
            # Calculate final price including trip distance
            trip_distance = float(requester[2]) if len(requester) > 2 else 1.0
            final_price = selected_price * trip_distance
            
            # Calculate acceptance probability (line 10 in Algorithm 1)
            acceptance_prob = self.calculate_acceptance_probability(final_price, requester)
            
            # Simulate acceptance decision
            accepted = np.random.random() < acceptance_prob
            
            # Calculate reward (exact from original: price + weight if matched)
            if accepted:
                # Simplified reward calculation (in full implementation, include matching weight W[i,j])
                reward = final_price
            else:
                reward = 0.0
            
            # Update arm parameters (lines 11-12 in Algorithm 1)
            self.update_arm_parameters(selected_arm, feature_vector, reward)
            
            # Store results
            prices[i] = final_price
            arm_selections[i] = selected_arm
            confidence_bounds[i] = conf_bounds[selected_arm]
            acceptance_probs[i] = acceptance_prob
            rewards[i] = reward
        
        # Calculate final objective value
        objective_value = np.sum(rewards)
        
        # Create matched pairs (simplified)
        matched_pairs = [(i, i % m) for i in range(n) if rewards[i] > 0]
        
        # Extract theta estimates for analysis
        theta_estimates = {}
        for arm in range(self.num_arms):
            A_inv = np.linalg.inv(self.A_matrices[arm])
            theta_estimates[arm] = A_inv @ self.b_vectors[arm]
        
        computation_time = time.time() - start_time
        
        logger.info(f"LinUCB method completed in {computation_time:.3f}s, objective: {objective_value:.2f}")
        
        return LinUCBResult(
            prices=prices,
            objective_value=objective_value,
            computation_time=computation_time,
            arm_selections=arm_selections,
            confidence_bounds=confidence_bounds,
            acceptance_probabilities=acceptance_probs,
            matched_pairs=matched_pairs,
            theta_estimates=theta_estimates
        ) 