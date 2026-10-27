"""
Server & Dashboard Subsystem.
Includes:
- app: The Flask application instance.
- start_server: Launcher function for the web server.
- SessionAnalytics: SCADA audit trail and performance analytics.
"""

from src.server.app import app, start_server, ml_engine, analytics
from src.server.analytics import SessionAnalytics, DEMO_MODES

__all__ = [
    "app",
    "start_server",
    "ml_engine",
    "analytics",
    "SessionAnalytics",
    "DEMO_MODES",
]
