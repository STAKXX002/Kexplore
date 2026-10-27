"""
Hardware Interface Subsystem.
Includes:
- run_bridge: USB Serial communication bridge for physical Arduino.
- parse_sensor_line: Telemetry line parser (JSON, comma-separated, key-value).
- auto_detect_port: Serial COM port auto-detector.
"""

from src.hardware.bridge import run_bridge, parse_sensor_line, auto_detect_port, list_available_ports

__all__ = [
    "run_bridge",
    "parse_sensor_line",
    "auto_detect_port",
    "list_available_ports",
]
