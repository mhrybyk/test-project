"""
Kalman Filter Implementation
Smooths position estimates and tracks velocity
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class KalmanFilter1D:
    """1D Kalman filter for single variable"""
    
    def __init__(
        self,
        process_variance: float,
        measurement_variance: float,
        initial_value: float = 0.0,
        initial_estimate_error: float = 1.0
    ):
        self.q = process_variance  # Process variance
        self.r = measurement_variance  # Measurement variance
        self.x = initial_value  # Initial value
        self.p = initial_estimate_error  # Initial estimate error
        self.initialized = False
    
    def update(self, measurement: float) -> float:
        """Update filter with new measurement"""
        
        # Prediction step
        self.p = self.p + self.q
        
        # Update step
        self.k = self.p / (self.p + self.r)  # Kalman gain
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        
        self.initialized = True
        return self.x
    
    def get_state(self) -> Tuple[float, float]:
        """Get current state and uncertainty"""
        return self.x, self.p


class KalmanFilter3D:
    """3D Kalman filter for position estimation"""
    
    def __init__(
        self,
        process_variance: float,
        measurement_variance: float,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        self.filters = [
            KalmanFilter1D(process_variance, measurement_variance, initial_position[i])
            for i in range(3)
        ]
    
    def update(self, measurement: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Update filter with new 3D measurement"""
        position = tuple(f.update(m) for f, m in zip(self.filters, measurement))
        return position
    
    def get_state(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get current position and uncertainties"""
        positions = tuple(f.x for f in self.filters)
        uncertainties = tuple(f.p for f in self.filters)
        return positions, uncertainties


class ExtendedKalmanFilter3D:
    """Extended Kalman Filter with velocity tracking"""
    
    def __init__(
        self,
        dt: float = 0.1,
        process_variance_position: float = 0.01,
        process_variance_velocity: float = 0.001,
        measurement_variance: float = 0.5,
        initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        """
        Initialize EKF with state vector: [x, y, z, vx, vy, vz]
        
        Args:
            dt: Time step
            process_variance_position: Process noise for position
            process_variance_velocity: Process noise for velocity
            measurement_variance: Measurement noise
            initial_position: Initial position estimate
        """
        
        self.dt = dt
        
        # State vector: [x, y, z, vx, vy, vz]
        self.x = np.array([
            initial_position[0], initial_position[1], initial_position[2],
            0.0, 0.0, 0.0
        ], dtype=float)
        
        # State transition matrix
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)
        
        # Process noise covariance
        self.Q = np.diag([
            process_variance_position,
            process_variance_position,
            process_variance_position,
            process_variance_velocity,
            process_variance_velocity,
            process_variance_velocity
        ])
        
        # Measurement matrix (we measure position only)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=float)
        
        # Measurement noise covariance
        self.R = np.eye(3) * measurement_variance
        
        # Estimate error covariance
        self.P = np.eye(6) * 1.0
    
    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: Tuple[float, float, float]):
        """Update step with new position measurement"""
        
        z = np.array(measurement, dtype=float)
        
        # Innovation
        y = z - (self.H @ self.x)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def get_position(self) -> Tuple[float, float, float]:
        """Get estimated position"""
        return tuple(self.x[:3])
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """Get estimated velocity"""
        return tuple(self.x[3:6])
    
    def get_state(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Get position and velocity"""
        return self.get_position(), self.get_velocity()
    
    def get_covariance(self) -> np.ndarray:
        """Get state covariance matrix"""
        return self.P


class IMMFilter:
    """
    Interacting Multiple Model filter
    Handles different motion models (static, constant velocity, acceleration)
    """
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.models = {
            'static': ExtendedKalmanFilter3D(
                dt=dt,
                process_variance_position=0.001,
                process_variance_velocity=0.0001,
                measurement_variance=0.5
            ),
            'constant_velocity': ExtendedKalmanFilter3D(
                dt=dt,
                process_variance_position=0.01,
                process_variance_velocity=0.001,
                measurement_variance=0.5
            ),
            'accelerated': ExtendedKalmanFilter3D(
                dt=dt,
                process_variance_position=0.1,
                process_variance_velocity=0.01,
                measurement_variance=0.5
            )
        }
        
        # Model probabilities
        self.model_probs = {'static': 0.33, 'constant_velocity': 0.33, 'accelerated': 0.33}
        
        # Markov chain transition matrix
        self.transition = np.array([
            [0.9, 0.05, 0.05],  # From static
            [0.1, 0.8, 0.1],    # From constant velocity
            [0.05, 0.1, 0.85]   # From accelerated
        ])
    
    def predict(self):
        """Prediction for all models"""
        for model in self.models.values():
            model.predict()
    
    def update(self, measurement: Tuple[float, float, float]):
        """Update all models and recalculate model probabilities"""
        
        model_names = list(self.models.keys())
        
        # Calculate likelihood for each model (simplified)
        likelihoods = {}
        for name, model in self.models.items():
            pos = model.get_position()
            error = np.linalg.norm(np.array(measurement) - np.array(pos))
            # Likelihood based on measurement error
            likelihood = np.exp(-error**2 / (2 * 0.5))
            likelihoods[name] = likelihood
        
        # Update model probabilities using Bayes rule
        total_likelihood = sum(likelihoods.values())
        for name in model_names:
            self.model_probs[name] = (likelihoods[name] * self.model_probs[name]) / (total_likelihood + 1e-10)
        
        # Normalize
        total_prob = sum(self.model_probs.values())
        for name in model_names:
            self.model_probs[name] /= total_prob
        
        # Update each model
        for model in self.models.values():
            model.update(measurement)
    
    def get_estimate(self) -> Tuple[Tuple[float, float, float], str]:
        """Get weighted position estimate and best model"""
        
        position = np.array([0.0, 0.0, 0.0])
        for name, model in self.models.items():
            pos = np.array(model.get_position())
            position += self.model_probs[name] * pos
        
        best_model = max(self.model_probs, key=self.model_probs.get)
        
        return tuple(position), best_model
