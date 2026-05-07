"""
UWB Protocol Implementation
Ultra-Wideband communication for ranging
"""

import serial
import struct
import time
import logging
import threading
from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class UWBMessageType(Enum):
    """UWB message types"""
    RANGING_REQUEST = 0x01
    RANGING_RESPONSE = 0x02
    RANGING_COMPLETE = 0x03
    CONFIG = 0x04
    STATUS = 0x05


@dataclass
class RangingMeasurement:
    """Single ranging measurement"""
    anchor_id: str
    tag_id: str
    distance: float  # meters
    rssi: float  # dBm
    timestamp: float
    measurement_quality: float  # 0-1


class UWBDevice:
    """Base UWB device class"""
    
    def __init__(
        self,
        device_id: str,
        serial_port: str,
        baud_rate: int = 115200,
        timeout: float = 1.0
    ):
        """
        Initialize UWB device
        
        Args:
            device_id: Unique device identifier
            serial_port: Serial port (e.g., '/dev/ttyUSB0')
            baud_rate: Serial baud rate
            timeout: Serial read timeout
        """
        
        self.device_id = device_id
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial = None
        self.connected = False
        self.last_ranging = {}
    
    def connect(self) -> bool:
        """Connect to UWB device via serial"""
        try:
            self.serial = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.connected = True
            logger.info(f"Connected to UWB device {self.device_id} on {self.serial_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.device_id}: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from UWB device"""
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.connected = False
        logger.info(f"Disconnected from UWB device {self.device_id}")
    
    def send_command(self, command: bytes) -> bool:
        """Send command to device"""
        if not self.connected:
            logger.error(f"Device {self.device_id} not connected")
            return False
        
        try:
            self.serial.write(command)
            return True
        except Exception as e:
            logger.error(f"Failed to send command to {self.device_id}: {e}")
            return False
    
    def read_response(self, expected_length: Optional[int] = None) -> Optional[bytes]:
        """Read response from device"""
        if not self.connected:
            return None
        
        try:
            if expected_length:
                data = self.serial.read(expected_length)
            else:
                data = self.serial.read(self.serial.in_waiting or 1)
            
            return data if data else None
        except Exception as e:
            logger.error(f"Failed to read from {self.device_id}: {e}")
            return None


class UWBAnchor(UWBDevice):
    """UWB Anchor device (stationary reference)"""
    
    def __init__(
        self,
        anchor_id: str,
        serial_port: str,
        position: Tuple[float, float, float],
        **kwargs
    ):
        """
        Initialize UWB anchor
        
        Args:
            anchor_id: Anchor identifier
            serial_port: Serial port
            position: 3D position (x, y, z)
            **kwargs: Additional arguments for UWBDevice
        """
        
        super().__init__(anchor_id, serial_port, **kwargs)
        self.position = position
        self.measurements = {}
    
    def request_ranging(self, tag_id: str) -> Optional[RangingMeasurement]:
        """Request ranging measurement from tag"""
        
        try:
            # Build ranging request
            command = self._build_ranging_request(tag_id)
            
            if not self.send_command(command):
                return None
            
            # Wait for response
            time.sleep(0.01)
            response = self.read_response(16)
            
            if response is None:
                return None
            
            # Parse response
            measurement = self._parse_ranging_response(response, tag_id)
            
            if measurement:
                self.last_ranging[tag_id] = measurement
                self.measurements[tag_id] = measurement
            
            return measurement
        
        except Exception as e:
            logger.error(f"Ranging error on {self.device_id}: {e}")
            return None
    
    def _build_ranging_request(self, tag_id: str) -> bytes:
        """Build ranging request command"""
        # Example format: [MSG_TYPE, ANCHOR_ID, TAG_ID, CRC]
        msg_type = UWBMessageType.RANGING_REQUEST.value
        anchor_bytes = self.device_id.encode()[:2]
        tag_bytes = tag_id.encode()[:2]
        crc = 0  # Simplified
        
        return bytes([msg_type]) + anchor_bytes + tag_bytes + bytes([crc])
    
    def _parse_ranging_response(
        self,
        response: bytes,
        tag_id: str
    ) -> Optional[RangingMeasurement]:
        """Parse ranging response"""
        
        try:
            if len(response) < 16:
                return None
            
            # Example parsing: [MSG_TYPE, STATUS, DISTANCE (float), RSSI, QUALITY]
            msg_type = response[0]
            status = response[1]
            
            if status != 0:
                return None
            
            # Parse distance (32-bit float)
            distance = struct.unpack('<f', response[2:6])[0]
            
            # Parse RSSI (signed byte)
            rssi = struct.unpack('b', response[6:7])[0]
            
            # Parse quality (byte 0-255 mapped to 0-1)
            quality = response[7] / 255.0
            
            return RangingMeasurement(
                anchor_id=self.device_id,
                tag_id=tag_id,
                distance=distance,
                rssi=rssi,
                timestamp=time.time(),
                measurement_quality=quality
            )
        
        except Exception as e:
            logger.error(f"Failed to parse ranging response: {e}")
            return None


class UWBTag(UWBDevice):
    """UWB Tag device (mobile/tracked)"""
    
    def __init__(
        self,
        tag_id: str,
        serial_port: str,
        **kwargs
    ):
        """
        Initialize UWB tag
        
        Args:
            tag_id: Tag identifier
            serial_port: Serial port
            **kwargs: Additional arguments for UWBDevice
        """
        
        super().__init__(tag_id, serial_port, **kwargs)
        self.position: Optional[Tuple[float, float, float]] = None
        self.velocity: Optional[Tuple[float, float, float]] = None
        self.last_update = time.time()
    
    def handle_ranging_request(self) -> Optional[bytes]:
        """Handle incoming ranging request and respond"""
        
        try:
            request = self.read_response(8)
            
            if request is None:
                return None
            
            # Build response
            response = self._build_ranging_response(request)
            
            if not self.send_command(response):
                return None
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling ranging request on {self.device_id}: {e}")
            return None
    
    def _build_ranging_response(self, request: bytes) -> bytes:
        """Build ranging response"""
        
        # Calculate response time (simulate Two-Way Ranging)
        response_time = time.time() % 256  # Simplified
        
        # Distance simulation (should be actual TWR calculation)
        distance = 5.0 + (time.time() % 1.0)  # Simplified
        rssi = -50  # Typical RSSI
        quality = 200  # Out of 255
        
        response = bytes([
            UWBMessageType.RANGING_RESPONSE.value,
            0,  # Status OK
        ]) + struct.pack('<f', distance) + bytes([
            rssi & 0xFF,
            quality
        ])
        
        return response
    
    def update_position(self, position: Tuple[float, float, float]):
        """Update tag position"""
        
        if self.position:
            # Calculate velocity
            dt = time.time() - self.last_update
            if dt > 0:
                self.velocity = tuple(
                    (position[i] - self.position[i]) / dt
                    for i in range(3)
                )
        
        self.position = position
        self.last_update = time.time()


class UWBRangingEngine:
    """Manages UWB ranging between anchors and tags"""
    
    def __init__(self):
        self.anchors: Dict[str, UWBAnchor] = {}
        self.tags: Dict[str, UWBTag] = {}
        self.measurements_callbacks: list = []
        self.running = False
    
    def add_anchor(self, anchor: UWBAnchor):
        """Add anchor to system"""
        self.anchors[anchor.device_id] = anchor
        logger.info(f"Added anchor {anchor.device_id}")
    
    def add_tag(self, tag: UWBTag):
        """Add tag to system"""
        self.tags[tag.device_id] = tag
        logger.info(f"Added tag {tag.device_id}")
    
    def connect_all(self) -> bool:
        """Connect all devices"""
        
        success = True
        
        for anchor in self.anchors.values():
            if not anchor.connect():
                success = False
        
        for tag in self.tags.values():
            if not tag.connect():
                success = False
        
        return success
    
    def disconnect_all(self):
        """Disconnect all devices"""
        
        for anchor in self.anchors.values():
            anchor.disconnect()
        
        for tag in self.tags.values():
            tag.disconnect()
    
    def perform_ranging(self, tag_id: str) -> Dict[str, RangingMeasurement]:
        """Perform ranging from all anchors to tag"""
        
        measurements = {}
        
        for anchor_id, anchor in self.anchors.items():
            measurement = anchor.request_ranging(tag_id)
            
            if measurement:
                measurements[anchor_id] = measurement
                
                # Call registered callbacks
                for callback in self.measurements_callbacks:
                    callback(measurement)
        
        return measurements
    
    def start_continuous_ranging(self, tag_id: str, interval: float = 0.1):
        """Start continuous ranging for tag"""
        
        self.running = True
        
        def ranging_loop():
            while self.running:
                self.perform_ranging(tag_id)
                time.sleep(interval)
        
        thread = threading.Thread(target=ranging_loop, daemon=True)
        thread.start()
    
    def stop_continuous_ranging(self):
        """Stop continuous ranging"""
        self.running = False
    
    def register_measurement_callback(self, callback: Callable):
        """Register callback for measurements"""
        self.measurements_callbacks.append(callback)
    
    def get_anchor_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all anchor positions"""
        return {
            anchor_id: anchor.position
            for anchor_id, anchor in self.anchors.items()
        }
    
    def get_tag_positions(self) -> Dict[str, Optional[Tuple[float, float, float]]]:
        """Get all tag positions"""
        return {
            tag_id: tag.position
            for tag_id, tag in self.tags.items()
        }
