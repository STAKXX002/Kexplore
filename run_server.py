"""
Starts the central Flask controller & web dashboard.
Usage:
    python run_server.py
"""

from src.server.app import start_server

if __name__ == '__main__':
    start_server()
