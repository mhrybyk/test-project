# UWB Indoor Positioning System

A complete, production-ready **Ultra-Wideband (UWB)** indoor positioning system with advanced filtering and trilateration algorithms.

## Overview

This system provides real-time indoor positioning using multiple UWB anchors and mobile tags. It includes:

- **UWB Protocol Layer**: Serial communication with UWB devices
- **Trilateration Engine**: 3D position calculation with outlier removal
- **Advanced Filtering**: Kalman filters with velocity tracking
- **REST API**: Complete HTTP interface for system control
- **Multi-threaded**: Continuous real-time positioning

## Architecture

```
UWB Devices (Anchors + Tags)
        ↓
   Serial Protocol Layer
        ↓
  Ranging Measurements
        ↓
Outlier Detection & Removal
        ↓
  Weighted Trilateration
        ↓
 Position Refinement
        ↓
Kalman Filtering (with velocity)
        ↓
REST API / Output
```

## Installation

### Prerequisites
- Python 3.8+
- USB serial connections to UWB devices

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python examples.py

# Start server
python position_server.py
```

## Core Modules

### `uwb_protocol.py`
UWB device communication and ranging:
- **UWBDevice**: Base serial communication
- **UWBAnchor**: Stationary reference points (range to tags)
- **UWBTag**: Mobile devices being tracked
- **UWBRangingEngine**: Manages multi-threaded ranging

Example:
```python
# Create anchors
anchor = UWBAnchor(
    anchor_id='A1',
    serial_port='/dev/ttyUSB0',
    position=(0, 0, 0)
)

# Connect and range
anchor.connect()
measurement = anchor.request_ranging('TAG1')
print(f"Distance: {measurement.distance}m, Quality: {measurement.measurement_quality}")
```

### `trilateration.py`
Position calculation from ranging measurements:
- **Trilateration**: Basic 3D/2D least-squares
- **WeightedTrilateration**: Quality-weighted positioning
- **OutlierDetection**: MAD/IQR-based filtering
- **Multilateration**: Complete positioning engine

Example:
```python
from trilateration import Multilateration

multilateration = Multilateration()

# Calculate position from measurements
estimate = multilateration.calculate_position(
    anchor_positions={
        'A1': (0, 0, 0),
        'A2': (10, 0, 0),
        'A3': (5, 8.66, 0),
        'A4': (5, 2.89, 3)
    },
    measurements={
        'A1': (5.2, 0.95),  # (distance, quality)
        'A2': (7.1, 0.92),
        'A3': (6.8, 0.88),
        'A4': (4.5, 0.90)
    },
    timestamp=time.time()
)

print(f"Position: {estimate.position}")
print(f"Quality: {estimate.quality}")
```

### `kalman_filter.py`
Advanced filtering algorithms:
- **KalmanFilter1D/3D**: Basic Kalman filters
- **ExtendedKalmanFilter3D**: Constant velocity model with state covariance
- **IMMFilter**: Adaptive model selection (static/constant velocity/accelerated)

Example:
```python
from kalman_filter import ExtendedKalmanFilter3D

# Create filter
kf = ExtendedKalmanFilter3D(dt=0.1)

# Process measurements
for measurement in measurements:
    kf.predict()
    kf.update(measurement)
    
    position, velocity = kf.get_state()
    print(f"Position: {position}, Velocity: {velocity}")
```

### `position_server.py`
Flask REST API server for system control and monitoring:

## REST API

### System Control

```
GET  /api/system/status          - System status
POST /api/system/start           - Start positioning
POST /api/system/stop            - Stop positioning
POST /api/calibrate              - Calibrate system
```

### Anchors

```
GET  /api/anchors                - Get all anchors
GET  /api/anchors/<id>           - Get specific anchor
```

### Tags & Positions

```
GET  /api/tags                   - Get all tags
GET  /api/tags/<id>/position     - Get tag position
GET  /api/tags/<id>/history      - Get position history
GET  /api/positions              - Get all current positions
```

### Configuration

```
GET  /api/config                 - Get configuration
POST /api/config                 - Update configuration
GET  /api/health                 - Health check
```

## Configuration

Edit `config.json` to configure anchors, filters, and API settings:

```json
{
  "system": {
    "update_rate": 10,
    "filter_enabled": true,
    "debug_mode": false
  },
  "anchors": [
    {
      "id": "A1",
      "position": [0.0, 0.0, 0.0],
      "serial_port": "/dev/ttyUSB0"
    },
    {
      "id": "A2",
      "position": [10.0, 0.0, 0.0],
      "serial_port": "/dev/ttyUSB1"
    }
  ],
  "kalman_filter": {
    "process_variance": 0.01,
    "measurement_variance": 0.5
  },
  "trilateration": {
    "min_anchors": 4,
    "max_distance": 100.0,
    "outlier_threshold": 2.0
  }
}
```

## Usage Examples

### Example 1: Basic Trilateration
```python
python -c "from examples import example_1_basic_trilateration; example_1_basic_trilateration()"
```

### Example 2: Weighted Positioning
```python
python -c "from examples import example_2_weighted_trilateration; example_2_weighted_trilateration()"
```

### Example 3: Complete System
```python
python examples.py
```

### Example 4: Run Server
```bash
python position_server.py

# In another terminal:
curl http://localhost:5000/api/system/status
curl -X POST http://localhost:5000/api/system/start
curl http://localhost:5000/api/positions
```

## Algorithm Details

### Trilateration (Least Squares)
Converts distance measurements to 3D position using least-squares optimization:

1. Build system matrix from anchor positions
2. Solve using SVD decomposition
3. Refine using gradient descent
4. Validate with residual checking

### Outlier Detection
- **MAD (Median Absolute Deviation)**: Robust to outliers
- **IQR (Interquartile Range)**: Standard statistical method

### Kalman Filtering
- **Standard 1D/3D**: Independent axis filtering
- **Extended KF**: Constant velocity model with state covariance
- **IMM Filter**: Multi-model adaptive filtering

## Performance

- **Position Accuracy**: ±0.1m to ±0.5m (depending on environment)
- **Update Rate**: 5-20 Hz
- **Latency**: 50-200ms
- **Multi-tag Support**: 10+ simultaneous tags

## Troubleshooting

### Low Position Accuracy
- Ensure minimum 4 anchors for 3D positioning
- Check anchor calibration
- Verify measurement quality scores
- Increase Kalman filter measurement variance

### Measurements Not Received
- Check serial port connections
- Verify baud rate settings
- Test UWB devices with vendor software first
- Check device power and antenna connections

### Position Jumps
- Increase Kalman filter process variance
- Use IMM filter for motion model adaptation
- Check for antenna multipath/reflections
- Verify anchor synchronization

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Minimum Anchors | 4 (for 3D) |
| Update Rate | 0.1-20 Hz |
| Operating Range | 10-100 meters |
| Position Accuracy | ±0.1-0.5m |
| Frequency Band | 3.1-10.6 GHz |
| Bandwidth | Up to 1.5 GHz |

## Dependencies

- **Flask**: REST API framework
- **NumPy**: Numerical computations
- **SciPy**: Optimization algorithms
- **PySerial**: Serial communication

## License

Open source - modify and use freely.

## Support

For issues or questions:
1. Check the examples in `examples.py`
2. Review configuration in `config.json`
3. Check logs in `position_server.py`
4. Verify UWB device connections and settings

## Contributing

Contributions welcome! Areas for enhancement:
- NLOS (Non-Line-of-Sight) mitigation
- Machine learning position refinement
- Web dashboard for visualization
- Additional filter types
- Performance optimization

---

**UWB Indoor Positioning System** - Real-time indoor tracking made simple.
