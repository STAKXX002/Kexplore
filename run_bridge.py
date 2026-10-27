"""
Starts the Arduino hardware communication bridge.
Usage:
    python run_bridge.py [--port COM3] [--baud 9600]
"""

import argparse
from src.hardware.bridge import run_bridge
from src.config import SERVER_URL, DEFAULT_BAUD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arduino Hardware Bridge")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default 9600)")
    parser.add_argument("--url", type=str, default=SERVER_URL, help="Server URL")
    args = parser.parse_args()
    run_bridge(port=args.port, baud=args.baud, server_url=args.url)
