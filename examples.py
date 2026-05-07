"""
Comprehensive Examples for UWB Indoor Positioning System
Demonstrates all major features and capabilities
"""

import time
import numpy as np
from typing import Dict, Tuple
import logging

# Import core modules
from trilateration import Trilateration, WeightedTrilateration, OutlierDetection, Multilateration
from kalman_filter import KalmanFilter3D, ExtendedKalmanFilter3D, IMMFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_basic_trilateration():
    """Example 1: Basic 3D Trilateration"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic 3D Trilateration")
    print("="*60)
    
    # Define anchor positions (meters)
    anchor_positions = {
        'A1': (0.0, 0.0, 0.0),
        'A2': (10.0, 0.0, 0.0),
        'A3': (5.0, 8.66, 0.0),
        'A4': (5.0, 2.89, 3.0)
    }
    
    # Simulated distances to tag (meters)
    distances = {
        'A1': 5.0,
        'A2': 7.5,
        'A3': 6.0,
        'A4': 4.5
    }
    
    print(f"\nAnchor Positions:")
    for aid, pos in anchor_positions.items():
        print(f"  {aid}: {pos}")
    
    print(f"\nMeasured Distances:")
    for aid, dist in distances.items():
        print(f"  {aid}: {dist:.2f}m")
    
    # Calculate position
    position = Trilateration.trilaterate_3d(anchor_positions, distances)
    
    if position:
        print(f"\n✓ Calculated Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    else:
        print("\n✗ Trilateration failed")


def example_2_weighted_trilateration():
    """Example 2: Trilateration with Quality Weighting"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Weighted Trilateration (Quality-Based)")
    print("="*60)
    
    anchor_positions = {
        'A1': (0.0, 0.0, 0.0),
        'A2': (10.0, 0.0, 0.0),
        'A3': (5.0, 8.66, 0.0),
        'A4': (5.0, 2.89, 3.0)
    }
    
    # Measurements with quality scores (0-1)
    measurements = {
        'A1': (5.0, 0.95),   # Good quality
        'A2': (7.5, 0.70),   # Poor quality
        'A3': (6.0, 0.90),   # Good quality
        'A4': (4.5, 0.85)    # Fair quality
    }
    
    print(f"\nMeasurements with Quality Scores:")
    for aid, (dist, quality) in measurements.items():
        print(f"  {aid}: {dist:.2f}m (quality: {quality:.0%})")
    
    # Weighted trilateration
    position = WeightedTrilateration.trilaterate_3d_weighted(anchor_positions, measurements)
    
    if position:
        print(f"\n✓ Weighted Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    else:
        print("\n✗ Weighted trilateration failed")


def example_3_outlier_detection():
    """Example 3: Outlier Detection and Removal"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Outlier Detection")
    print("="*60)
    
    # Measurements with one outlier
    measurements = {
        'A1': (5.0, 0.95),
        'A2': (7.5, 0.90),
        'A3': (6.0, 0.85),
        'A4': (25.0, 0.60)    # OUTLIER!
    }
    
    print(f"\nOriginal measurements:")
    for aid, (dist, quality) in measurements.items():
        print(f"  {aid}: {dist:.2f}m (quality: {quality:.0%})")
    
    # Detect outliers using MAD method
    distances = {aid: m[0] for aid, m in measurements.items()}
    outliers_mad = OutlierDetection.mad_based_outliers(distances, threshold=2.0)
    
    print(f"\nOutlier Detection (MAD method):")
    for aid, is_outlier in outliers_mad.items():
        status = "⚠ OUTLIER" if is_outlier else "✓ Valid"
        print(f"  {aid}: {status}")
    
    # Remove outliers
    filtered = OutlierDetection.remove_outliers(measurements, method='mad')
    
    print(f"\nFiltered measurements ({len(filtered)}/{len(measurements)} valid):")
    for aid, (dist, quality) in filtered.items():
        print(f"  {aid}: {dist:.2f}m")


def example_4_kalman_filtering():
    """Example 4: Kalman Filter for Position Smoothing"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Kalman Filter Position Smoothing")
    print("="*60)
    
    # Create 3D Kalman filter
    kf = KalmanFilter3D(
        process_variance=0.01,
        measurement_variance=0.5,
        initial_position=(0.0, 0.0, 0.0)
    )
    
    # Simulate noisy measurements
    true_position = np.array([5.0, 5.0, 2.0])
    noise_std = 0.3
    
    print(f"\nTrue position: {true_position}")
    print(f"Noise std: {noise_std}")
    print(f"\n{'Time':<6} {'Noisy Measurement':<20} {'Filtered Position':<20}")
    print("-" * 50)
    
    for t in range(1, 11):
        # Generate noisy measurement
        measurement = true_position + np.random.normal(0, noise_std, 3)
        
        # Update filter
        filtered = kf.update(tuple(measurement))
        
        print(f"{t:<6} {str(tuple(measurement.round(2))):<20} {str(tuple(np.array(filtered).round(2))):<20}")
    
    print(f"\n✓ Filter converged to: {filtered}")


def example_5_extended_kalman_filter():
    """Example 5: Extended Kalman Filter with Velocity Tracking"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Extended Kalman Filter (Velocity Tracking)")
    print("="*60)
    
    ekf = ExtendedKalmanFilter3D(
        dt=0.1,
        process_variance_position=0.01,
        process_variance_velocity=0.001,
        measurement_variance=0.5
    )
    
    print(f"\nSimulating object moving at constant velocity...")
    print(f"Velocity: (1.0, 0.5, 0.2) m/s")
    print(f"\n{'Time':<6} {'Measured Pos':<25} {'Est. Pos':<25} {'Est. Velocity':<25}")
    print("-" * 80)
    
    # Simulate object moving with constant velocity
    true_pos = np.array([0.0, 0.0, 0.0])
    velocity = np.array([1.0, 0.5, 0.2])
    
    for t in range(10):
        # True position
        true_pos = true_pos + velocity * 0.1
        
        # Noisy measurement
        measurement = true_pos + np.random.normal(0, 0.2, 3)
        
        # Predict and update
        ekf.predict()
        ekf.update(tuple(measurement))
        
        est_pos, est_vel = ekf.get_state()
        
        print(f"{t:<6} {str(tuple(measurement.round(2))):<25} {str(tuple(np.array(est_pos).round(2))):<25} {str(tuple(np.array(est_vel).round(2))):<25}")
    
    final_pos, final_vel = ekf.get_state()
    print(f"\n✓ Final velocity estimate: {np.array(final_vel).round(3)}")


def example_6_imm_filter():
    """Example 6: Interacting Multiple Model Filter"""
    print("\n" + "="*60)
    print("EXAMPLE 6: IMM Filter (Adaptive Motion Models)")
    print("="*60)
    
    imm = IMMFilter(dt=0.1)
    
    print(f"\nTesting IMM with different motion patterns...")
    print(f"\n{'Phase':<15} {'Model':<20} {'Position':<25} {'Probability':<15}")
    print("-" * 80)
    
    # Phase 1: Static (no motion)
    pos = np.array([5.0, 5.0, 2.0])
    for i in range(5):
        imm.predict()
        imm.update(tuple(pos + np.random.normal(0, 0.1, 3)))
        est_pos, best_model = imm.get_estimate()
        prob = imm.model_probs[best_model]
        print(f"{'Static':<15} {best_model:<20} {str(tuple(np.array(est_pos).round(2))):<25} {prob:.1%}")
    
    # Phase 2: Constant velocity
    print()
    velocity = np.array([1.0, 0.5, 0.0])
    for i in range(5):
        pos = pos + velocity * 0.1
        imm.predict()
        imm.update(tuple(pos + np.random.normal(0, 0.1, 3)))
        est_pos, best_model = imm.get_estimate()
        prob = imm.model_probs[best_model]
        print(f"{'Constant Vel':<15} {best_model:<20} {str(tuple(np.array(est_pos).round(2))):<25} {prob:.1%}")
    
    print(f"\n✓ IMM successfully adapted to different motion models")


def example_7_complete_system():
    """Example 7: Complete Positioning System"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Complete Positioning System")
    print("="*60)
    
    # Initialize system
    multilateration = Multilateration()
    ekf = ExtendedKalmanFilter3D(dt=0.1)
    
    # Anchor configuration
    anchor_positions = {
        'A1': (0.0, 0.0, 0.0),
        'A2': (10.0, 0.0, 0.0),
        'A3': (5.0, 8.66, 0.0),
        'A4': (5.0, 2.89, 3.0)
    }
    
    print(f"\nSystem Configuration:")
    print(f"  Anchors: {len(anchor_positions)}")
    print(f"  Anchor coverage area: 10m x 8.66m x 3m")
    
    # Simulate continuous positioning
    print(f"\nSimulating 20 position measurements...")
    print(f"\n{'Iter':<5} {'Position':<25} {'Quality':<10} {'Residual':<10} {'Filtered':<25}")
    print("-" * 80)
    
    # Simulate tag moving in a circle
    true_pos = np.array([5.0, 4.0, 1.5])
    
    for iteration in range(20):
        # Update true position (circular motion)
        angle = iteration * 2 * np.pi / 20
        true_pos = np.array([5.0 + 2.0 * np.cos(angle), 4.0 + 1.5 * np.sin(angle), 1.5])
        
        # Generate measurements from all anchors
        measurements = {}
        for aid, anchor_pos in anchor_positions.items():
            distance = np.linalg.norm(true_pos - np.array(anchor_pos))
            quality = 0.90 + 0.08 * np.random.random()  # Random quality 0.90-0.98
            measurements[aid] = (distance, quality)
        
        # Calculate position
        estimate = multilateration.calculate_position(anchor_positions, measurements, time.time())
        
        if estimate:
            # Apply Kalman filter
            ekf.predict()
            ekf.update(estimate.position)
            filtered_pos, _ = ekf.get_state()
            
            print(f"{iteration+1:<5} {str(tuple(estimate.position))[:25]:<25} {estimate.quality:.2%} {estimate.residual:.3f}      {str(tuple(np.array(filtered_pos).round(2)))[:25]:<25}")
        else:
            print(f"{iteration+1:<5} Failed to calculate position")
    
    # Statistics
    history = multilateration.get_position_history(20)
    if history:
        avg_quality = np.mean([est.quality for est in history])
        avg_residual = np.mean([est.residual for est in history])
        print(f"\n✓ System Statistics:")
        print(f"  Average quality: {avg_quality:.2%}")
        print(f"  Average residual: {avg_residual:.4f}m")
        print(f"  Position history: {len(history)} estimates")


def run_all_examples():
    """Run all examples"""
    print("\n\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  UWB INDOOR POSITIONING SYSTEM - EXAMPLES".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    try:
        example_1_basic_trilateration()
        example_2_weighted_trilateration()
        example_3_outlier_detection()
        example_4_kalman_filtering()
        example_5_extended_kalman_filter()
        example_6_imm_filter()
        example_7_complete_system()
        
        print("\n\n" + "="*60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == '__main__':
    run_all_examples()
