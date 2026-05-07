"""
Trilateration and Position Calculation
Converts ranging measurements to position estimates
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionEstimate:
    """Position estimate with quality metrics"""
    position: Tuple[float, float, float]
    timestamp: float
    num_anchors: int
    residual: float
    quality: float  # 0-1
    variance: Optional[np.ndarray] = None


class Trilateration:
    """3D/2D trilateration using least-squares"""
    
    @staticmethod
    def trilaterate_3d(
        anchor_positions: Dict[str, Tuple[float, float, float]],
        distances: Dict[str, float],
        initial_guess: Optional[Tuple[float, float, float]] = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        3D trilateration using least-squares optimization
        
        Args:
            anchor_positions: Dict of anchor_id -> (x, y, z)
            distances: Dict of anchor_id -> distance
            initial_guess: Initial position estimate for optimization
        
        Returns:
            Position (x, y, z) or None if failed
        """
        
        if len(anchor_positions) < 4:
            logger.warning("Need at least 4 anchors for 3D trilateration")
            return None
        
        # Prepare matrices for least-squares
        anchors = list(anchor_positions.keys())
        P = np.array([anchor_positions[a] for a in anchors])
        d = np.array([distances[a] for a in anchors])
        
        # Use first anchor as reference
        P_ref = P[0]
        P_rel = P[1:] - P_ref
        d_rel = d[1:] - d[0]
        
        # Build system: A * x = b
        A = 2 * P_rel
        
        # Compute b
        sum_P_sq = np.sum(P**2, axis=1)
        b = (d[0]**2 - sum_P_sq[0]) - (d[1:]**2 - sum_P_sq[1:]) + 2 * np.dot(P_rel, P_ref)
        
        try:
            # Solve using least-squares
            x_solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            
            position = tuple(x_solution)
            return position
        
        except np.linalg.LinAlgError as e:
            logger.error(f"Trilateration failed: {e}")
            return None
    
    @staticmethod
    def trilaterate_2d(
        anchor_positions: Dict[str, Tuple[float, float]],
        distances: Dict[str, float]
    ) -> Optional[Tuple[float, float]]:
        """
        2D trilateration for planar positioning
        
        Args:
            anchor_positions: Dict of anchor_id -> (x, y)
            distances: Dict of anchor_id -> distance
        
        Returns:
            Position (x, y) or None if failed
        """
        
        if len(anchor_positions) < 3:
            logger.warning("Need at least 3 anchors for 2D trilateration")
            return None
        
        anchors = list(anchor_positions.keys())
        P = np.array([anchor_positions[a] for a in anchors])
        d = np.array([distances[a] for a in anchors])
        
        # Use first anchor as reference
        P_ref = P[0]
        P_rel = P[1:] - P_ref
        
        # Build system
        A = 2 * P_rel
        sum_P_sq = np.sum(P**2, axis=1)
        b = (d[0]**2 - sum_P_sq[0]) - (d[1:]**2 - sum_P_sq[1:]) + 2 * np.dot(P_rel, P_ref)
        
        try:
            x_solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            position = tuple(x_solution)
            return position
        
        except np.linalg.LinAlgError as e:
            logger.error(f"2D trilateration failed: {e}")
            return None


class WeightedTrilateration:
    """Trilateration with weighted measurements"""
    
    @staticmethod
    def trilaterate_3d_weighted(
        anchor_positions: Dict[str, Tuple[float, float, float]],
        measurements: Dict[str, Tuple[float, float]],  # (distance, quality)
    ) -> Optional[Tuple[float, float, float]]:
        """
        3D trilateration with quality weighting
        
        Args:
            anchor_positions: Dict of anchor_id -> (x, y, z)
            measurements: Dict of anchor_id -> (distance, quality)
        
        Returns:
            Position (x, y, z) or None
        """
        
        if len(anchor_positions) < 4:
            return None
        
        anchors = list(anchor_positions.keys())
        P = np.array([anchor_positions[a] for a in anchors])
        distances = np.array([measurements[a][0] for a in anchors])
        qualities = np.array([measurements[a][1] for a in anchors])
        
        # Normalize weights
        weights = qualities / np.sum(qualities)
        
        # Weight the measurements
        P_ref = P[0]
        P_rel = P[1:] - P_ref
        d_rel = distances[1:] - distances[0]
        
        # Weighted least-squares
        W = np.diag(weights[1:])
        A = 2 * P_rel
        
        sum_P_sq = np.sum(P**2, axis=1)
        b = (distances[0]**2 - sum_P_sq[0]) - (distances[1:]**2 - sum_P_sq[1:]) + 2 * np.dot(P_rel, P_ref)
        
        try:
            # Solve: (A^T W A)^-1 A^T W b
            ATA = A.T @ W @ A
            ATb = A.T @ W @ b
            x_solution = np.linalg.solve(ATA, ATb)
            return tuple(x_solution)
        
        except np.linalg.LinAlgError:
            return None


class OutlierDetection:
    """Outlier detection and removal"""
    
    @staticmethod
    def mad_based_outliers(
        distances: Dict[str, float],
        threshold: float = 2.0
    ) -> Dict[str, bool]:
        """
        Median Absolute Deviation (MAD) based outlier detection
        
        Args:
            distances: Dict of anchor_id -> distance
            threshold: MAD threshold (typically 2-3)
        
        Returns:
            Dict of anchor_id -> is_outlier
        """
        
        values = np.array(list(distances.values()))
        
        # Calculate median
        median = np.median(values)
        
        # Calculate MAD
        mad = np.median(np.abs(values - median))
        
        # Detect outliers
        outliers = {}
        for anchor_id, distance in distances.items():
            if mad > 0:
                z_score = (distance - median) / (1.4826 * mad)
                outliers[anchor_id] = abs(z_score) > threshold
            else:
                outliers[anchor_id] = False
        
        return outliers
    
    @staticmethod
    def iqr_based_outliers(
        distances: Dict[str, float],
        k: float = 1.5
    ) -> Dict[str, bool]:
        """
        IQR (Interquartile Range) based outlier detection
        
        Args:
            distances: Dict of anchor_id -> distance
            k: IQR multiplier (default 1.5)
        
        Returns:
            Dict of anchor_id -> is_outlier
        """
        
        values = np.array(list(distances.values()))
        
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        outliers = {}
        for anchor_id, distance in distances.items():
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            outliers[anchor_id] = (distance < lower_bound or distance > upper_bound)
        
        return outliers
    
    @staticmethod
    def remove_outliers(
        measurements: Dict[str, Tuple[float, float]],
        method: str = 'mad'
    ) -> Dict[str, Tuple[float, float]]:
        """
        Remove outlier measurements
        
        Args:
            measurements: Dict of anchor_id -> (distance, quality)
            method: 'mad' or 'iqr'
        
        Returns:
            Filtered measurements
        """
        
        distances = {aid: m[0] for aid, m in measurements.items()}
        
        if method == 'mad':
            outliers = OutlierDetection.mad_based_outliers(distances)
        else:
            outliers = OutlierDetection.iqr_based_outliers(distances)
        
        return {
            aid: measurements[aid]
            for aid in measurements.keys()
            if not outliers.get(aid, False)
        }


class PositionValidator:
    """Validates and refines position estimates"""
    
    @staticmethod
    def validate_position(
        position: Tuple[float, float, float],
        anchor_positions: Dict[str, Tuple[float, float, float]],
        measurements: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Validate position by checking residuals
        
        Returns:
            Tuple of (residual, quality_score 0-1)
        """
        
        residuals = []
        
        for anchor_id, anchor_pos in anchor_positions.items():
            if anchor_id not in measurements:
                continue
            
            # Calculate expected distance
            distance = np.linalg.norm(np.array(position) - np.array(anchor_pos))
            
            # Measured distance
            measured_distance = measurements[anchor_id][0]
            
            # Residual
            residual = abs(distance - measured_distance)
            residuals.append(residual)
        
        if not residuals:
            return float('inf'), 0.0
        
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Quality: lower residual = higher quality
        quality = 1.0 / (1.0 + mean_residual)
        
        return mean_residual, quality
    
    @staticmethod
    def refine_position(
        position: Tuple[float, float, float],
        anchor_positions: Dict[str, Tuple[float, float, float]],
        measurements: Dict[str, Tuple[float, float]],
        iterations: int = 5
    ) -> Tuple[float, float, float]:
        """
        Refine position estimate using gradient descent
        
        Args:
            position: Initial position
            anchor_positions: Anchor locations
            measurements: (distance, quality) measurements
            iterations: Number of refinement iterations
        
        Returns:
            Refined position
        """
        
        pos = np.array(position, dtype=float)
        learning_rate = 0.1
        
        for _ in range(iterations):
            gradient = np.zeros(3)
            
            for anchor_id, anchor_pos in anchor_positions.items():
                if anchor_id not in measurements:
                    continue
                
                anchor = np.array(anchor_pos)
                measured_dist = measurements[anchor_id][0]
                
                # Calculate current distance
                diff = pos - anchor
                current_dist = np.linalg.norm(diff)
                
                if current_dist > 0:
                    # Gradient towards correct distance
                    gradient += (current_dist - measured_dist) * (diff / current_dist)
            
            # Update position
            pos = pos - learning_rate * gradient
        
        return tuple(pos)


class Multilateration:
    """Complete multilateration engine"""
    
    def __init__(self):
        self.position_history: List[PositionEstimate] = []
        self.max_history = 1000
    
    def calculate_position(
        self,
        anchor_positions: Dict[str, Tuple[float, float, float]],
        measurements: Dict[str, Tuple[float, float]],
        timestamp: float
    ) -> Optional[PositionEstimate]:
        """
        Calculate position from measurements
        
        Args:
            anchor_positions: Dict of anchor -> position
            measurements: Dict of anchor -> (distance, quality)
            timestamp: Measurement timestamp
        
        Returns:
            PositionEstimate or None
        """
        
        if len(measurements) < 4:
            logger.warning(f"Need at least 4 measurements, got {len(measurements)}")
            return None
        
        # Remove outliers
        filtered_measurements = OutlierDetection.remove_outliers(measurements)
        
        if len(filtered_measurements) < 4:
            logger.warning("Not enough valid measurements after outlier removal")
            return None
        
        # Filter anchor positions to only those with measurements
        filtered_anchors = {
            aid: anchor_positions[aid]
            for aid in filtered_measurements.keys()
            if aid in anchor_positions
        }
        
        # Perform trilateration
        position = WeightedTrilateration.trilaterate_3d_weighted(
            filtered_anchors,
            filtered_measurements
        )
        
        if position is None:
            logger.error("Trilateration failed")
            return None
        
        # Refine position
        position = PositionValidator.refine_position(
            position,
            filtered_anchors,
            filtered_measurements
        )
        
        # Validate
        residual, quality = PositionValidator.validate_position(
            position,
            filtered_anchors,
            filtered_measurements
        )
        
        # Calculate variance
        variance = np.array([quality] * 3)  # Simplified
        
        estimate = PositionEstimate(
            position=position,
            timestamp=timestamp,
            num_anchors=len(filtered_measurements),
            residual=residual,
            quality=quality,
            variance=variance
        )
        
        # Add to history
        self.position_history.append(estimate)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
        
        return estimate
    
    def get_position_history(self, window: int = 10) -> List[PositionEstimate]:
        """Get last N position estimates"""
        return self.position_history[-window:]
    
    def get_average_position(self, window: int = 10) -> Optional[Tuple[float, float, float]]:
        """Get average position over window"""
        
        history = self.get_position_history(window)
        
        if not history:
            return None
        
        positions = np.array([est.position for est in history])
        avg_position = tuple(np.mean(positions, axis=0))
        
        return avg_position
