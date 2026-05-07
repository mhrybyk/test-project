"""
UWB Positioning Server
Flask REST API for system control and real-time positioning
"""

import json
import threading
import logging
from typing import Dict
from datetime import datetime
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from uwb_protocol import UWBAnchor, UWBTag, UWBRangingEngine, RangingMeasurement
from trilateration import Multilateration
from kalman_filter import ExtendedKalmanFilter3D

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global system state
system = {
    'running': False,
    'anchors': {},
    'tags': {},
    'engine': None,
    'multilateration': Multilateration(),
    'filters': {},
    'config': {}
}


def load_config(config_file: str = 'config.json'):
    """Load system configuration"""
    try:
        with open(config_file, 'r') as f:
            system['config'] = json.load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        system['config'] = {}


def initialize_system():
    """Initialize UWB system from config"""
    try:
        config = system['config']
        
        # Create ranging engine
        system['engine'] = UWBRangingEngine()
        
        # Create anchors
        for anchor_config in config.get('anchors', []):
            anchor = UWBAnchor(
                anchor_id=anchor_config['id'],
                serial_port=anchor_config['serial_port'],
                position=tuple(anchor_config['position'])
            )
            system['anchors'][anchor_config['id']] = anchor
            system['engine'].add_anchor(anchor)
        
        # Create Kalman filters for each tag
        for tag_config in config.get('anchors', []):
            tag_id = tag_config['id']
            kf_config = config.get('kalman_filter', {})
            system['filters'][tag_id] = ExtendedKalmanFilter3D(
                process_variance_position=kf_config.get('process_variance', 0.01),
                measurement_variance=kf_config.get('measurement_variance', 0.5)
            )
        
        # Register measurement callback
        system['engine'].register_measurement_callback(on_measurement)
        
        logger.info(f"System initialized with {len(system['anchors'])} anchors")
        return True
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return False


def on_measurement(measurement: RangingMeasurement):
    """Callback for ranging measurements"""
    logger.debug(f"Measurement: {measurement.anchor_id} -> {measurement.tag_id}: {measurement.distance:.2f}m")


def positioning_worker(tag_id: str, interval: float = 0.1):
    """Background worker for continuous positioning"""
    
    while system['running']:
        try:
            # Get measurements from all anchors
            measurements = system['engine'].perform_ranging(tag_id)
            
            if not measurements:
                time.sleep(interval)
                continue
            
            # Convert to (distance, quality) format
            measurement_dict = {
                anchor_id: (m.distance, m.measurement_quality)
                for anchor_id, m in measurements.items()
            }
            
            # Get anchor positions
            anchor_positions = system['engine'].get_anchor_positions()
            
            # Calculate position
            estimate = system['multilateration'].calculate_position(
                anchor_positions,
                measurement_dict,
                time.time()
            )
            
            if estimate:
                # Apply Kalman filter
                if tag_id in system['filters']:
                    kf = system['filters'][tag_id]
                    kf.predict()
                    kf.update(estimate.position)
                    filtered_pos, velocity = kf.get_state()
                    logger.info(f"Tag {tag_id}: pos={filtered_pos}, vel={velocity}, quality={estimate.quality:.2f}")
        
        except Exception as e:
            logger.error(f"Positioning worker error: {e}")
        
        time.sleep(interval)


# REST API Endpoints

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    return jsonify({
        'running': system['running'],
        'num_anchors': len(system['anchors']),
        'num_tags': len(system['tags']),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start positioning system"""
    try:
        if not system['anchors']:
            if not initialize_system():
                return jsonify({'error': 'Failed to initialize system'}), 500
        
        if not system['engine'].connect_all():
            return jsonify({'error': 'Failed to connect to devices'}), 500
        
        system['running'] = True
        
        # Start positioning workers for each tag
        for tag_id in system['tags']:
            thread = threading.Thread(
                target=positioning_worker,
                args=(tag_id, 0.1),
                daemon=True
            )
            thread.start()
        
        logger.info("System started")
        return jsonify({'status': 'started'})
    
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Stop positioning system"""
    try:
        system['running'] = False
        system['engine'].disconnect_all()
        logger.info("System stopped")
        return jsonify({'status': 'stopped'})
    
    except Exception as e:
        logger.error(f"Failed to stop system: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/anchors', methods=['GET'])
def get_anchors():
    """Get all anchors"""
    anchors = []
    for anchor_id, anchor in system['anchors'].items():
        anchors.append({
            'id': anchor_id,
            'position': anchor.position,
            'connected': anchor.connected
        })
    return jsonify(anchors)


@app.route('/api/anchors/<anchor_id>', methods=['GET'])
def get_anchor(anchor_id):
    """Get specific anchor"""
    if anchor_id not in system['anchors']:
        return jsonify({'error': 'Anchor not found'}), 404
    
    anchor = system['anchors'][anchor_id]
    return jsonify({
        'id': anchor_id,
        'position': anchor.position,
        'connected': anchor.connected,
        'measurements': len(anchor.measurements)
    })


@app.route('/api/tags', methods=['GET'])
def get_tags():
    """Get all tags"""
    tags = []
    for tag_id, tag in system['tags'].items():
        tags.append({
            'id': tag_id,
            'position': tag.position,
            'velocity': tag.velocity
        })
    return jsonify(tags)


@app.route('/api/tags/<tag_id>/position', methods=['GET'])
def get_tag_position(tag_id):
    """Get tag position"""
    if tag_id not in system['tags']:
        return jsonify({'error': 'Tag not found'}), 404
    
    tag = system['tags'][tag_id]
    
    # Get filtered position
    filtered_pos = None
    if tag_id in system['filters']:
        filtered_pos, _ = system['filters'][tag_id].get_state()
    
    return jsonify({
        'id': tag_id,
        'position': tag.position,
        'velocity': tag.velocity,
        'filtered_position': filtered_pos,
        'timestamp': tag.last_update
    })


@app.route('/api/tags/<tag_id>/history', methods=['GET'])
def get_tag_history(tag_id):
    """Get position history for tag"""
    window = request.args.get('window', 50, type=int)
    
    history = system['multilateration'].get_position_history(window)
    
    return jsonify({
        'tag_id': tag_id,
        'history': [
            {
                'position': est.position,
                'timestamp': est.timestamp,
                'quality': est.quality,
                'residual': est.residual,
                'num_anchors': est.num_anchors
            }
            for est in history
        ]
    })


@app.route('/api/positions', methods=['GET'])
def get_all_positions():
    """Get all current positions"""
    positions = {}
    
    for tag_id, tag in system['tags'].items():
        filtered_pos = None
        if tag_id in system['filters']:
            filtered_pos, _ = system['filters'][tag_id].get_state()
        
        positions[tag_id] = {
            'raw_position': tag.position,
            'filtered_position': filtered_pos,
            'velocity': tag.velocity
        }
    
    return jsonify(positions)


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration"""
    return jsonify(system['config'])


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update system configuration"""
    try:
        config = request.get_json()
        system['config'] = config
        
        # Save to file
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Configuration updated")
        return jsonify({'status': 'updated'})
    
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Calibrate system (placeholder)"""
    try:
        logger.info("Starting system calibration")
        # Add calibration logic here
        return jsonify({'status': 'calibration_started'})
    
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - API documentation"""
    return jsonify({
        'name': 'UWB Indoor Positioning System',
        'version': '1.0.0',
        'endpoints': {
            'system': {
                'GET /api/system/status': 'Get system status',
                'POST /api/system/start': 'Start positioning system',
                'POST /api/system/stop': 'Stop positioning system'
            },
            'anchors': {
                'GET /api/anchors': 'Get all anchors',
                'GET /api/anchors/<id>': 'Get specific anchor'
            },
            'tags': {
                'GET /api/tags': 'Get all tags',
                'GET /api/tags/<id>/position': 'Get tag position',
                'GET /api/tags/<id>/history': 'Get position history'
            },
            'positions': {
                'GET /api/positions': 'Get all current positions'
            },
            'configuration': {
                'GET /api/config': 'Get configuration',
                'POST /api/config': 'Update configuration'
            },
            'system': {
                'GET /api/health': 'Health check',
                'POST /api/calibrate': 'Calibrate system'
            }
        }
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main entry point"""
    
    # Load configuration
    load_config()
    
    # Initialize system
    initialize_system()
    
    # Start Flask server
    host = system['config'].get('api', {}).get('host', '0.0.0.0')
    port = system['config'].get('api', {}).get('port', 5000)
    debug = system['config'].get('api', {}).get('debug', False)
    
    logger.info(f"Starting UWB Positioning Server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
