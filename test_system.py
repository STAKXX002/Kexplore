"""
Automated End-to-End Verification Test Suite.
Tests:
1. Model loading and ML Engine predictions.
2. Receiver endpoints (/telemetry, /api/live-status, /api/toggle-mode, /dashboard).
3. Arduino bridge line parsing with various payload formats.
"""

import unittest
import json
from ml_engine import MLEngine
from receiver import app
from arduino_bridge import parse_sensor_line

class TestSystem(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.engine = MLEngine()

    def test_ml_engine_normal(self):
        res = self.engine.process_telemetry(27.0, 50.0, 27.2, 50.0)
        self.assertIn("predicted_temp", res)
        self.assertIn("is_anomaly", res)
        self.assertIn("led_1", res)
        self.assertIn("led_2", res)
        self.assertFalse(res["tripped"])

    def test_ml_engine_cross_sensor_anomaly(self):
        # Huge divergence between sensor 1 and sensor 2 (e.g. 25°C vs 35°C)
        res = self.engine.process_telemetry(25.0, 50.0, 35.0, 50.0)
        self.assertTrue(res["is_anomaly"])
        self.assertIn("Cross-Sensor Divergence", res["anomaly_reason"])

    def test_ml_engine_mode_toggle(self):
        self.assertTrue(self.engine.set_control_mode("CLASSICAL"))
        self.assertEqual(self.engine.control_mode, "CLASSICAL")
        self.assertTrue(self.engine.set_control_mode("AI_PREDICTIVE"))
        self.assertEqual(self.engine.control_mode, "AI_PREDICTIVE")

    def test_receiver_telemetry_endpoint(self):
        payload = {
            "sensor_1": 28.5,
            "sensor_2": 28.2,
            "humidity_1": 52.0,
            "humidity_2": 51.0
        }
        resp = self.app.post('/telemetry', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("control_signal", data)
        self.assertIn("ai_insights", data)

    def test_receiver_live_status_endpoint(self):
        resp = self.app.get('/api/live-status')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("latest", data)
        self.assertIn("history", data)

    def test_receiver_dashboard_endpoint(self):
        resp = self.app.get('/dashboard')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Dual-Sensor AI/ML Predictive IoT Controller", resp.data)

    def test_arduino_bridge_parsers(self):
        # Test JSON format
        j = parse_sensor_line('{"sensor_1": 29.1, "sensor_2": 28.8}')
        self.assertIsNotNone(j)
        self.assertEqual(j["sensor_1"], 29.1)
        self.assertEqual(j["sensor_2"], 28.8)

        # Test comma-separated format
        c = parse_sensor_line('27.4, 26.9')
        self.assertIsNotNone(c)
        self.assertEqual(c["sensor_1"], 27.4)
        self.assertEqual(c["sensor_2"], 26.9)

        # Test key-value format
        kv = parse_sensor_line('S1=30.0, S2=29.5')
        self.assertIsNotNone(kv)
        self.assertEqual(kv["sensor_1"], 30.0)
        self.assertEqual(kv["sensor_2"], 29.5)

if __name__ == '__main__':
    unittest.main()
