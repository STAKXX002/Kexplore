"""
Automated Test Suite for Multi-Modal AI System (XGBoost + PyTorch Autoencoder).
"""

import unittest
from ml_engine import MLEngine
from receiver import app
from arduino_bridge import parse_sensor_line

class TestMultiModalSystem(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.engine = MLEngine()

    def test_ml_engine_normal_telemetry(self):
        res = self.engine.process_telemetry(27.0, 55.0)
        self.assertIn("predicted_temp", res)
        self.assertIn("recon_loss", res)
        self.assertIn("is_anomaly", res)
        self.assertIn("led_1", res)
        self.assertIn("led_2", res)
        self.assertFalse(res["tripped"])

    def test_ml_engine_darkness_thermal_anomaly(self):
        # Abnormal: Dark room (5% light) with soaring heat (34°C)
        res = self.engine.process_telemetry(34.0, 5.0)
        self.assertTrue(res["is_anomaly"])
        self.assertIn("Thermal Runaway in Darkness", res["anomaly_reason"])

    def test_ml_engine_mode_toggle(self):
        self.assertTrue(self.engine.set_control_mode("CLASSICAL"))
        self.assertEqual(self.engine.control_mode, "CLASSICAL")
        self.assertTrue(self.engine.set_control_mode("AI_PREDICTIVE"))
        self.assertEqual(self.engine.control_mode, "AI_PREDICTIVE")

    def test_receiver_telemetry_endpoint(self):
        payload = {
            "temperature": 28.5,
            "light": 65.0
        }
        resp = self.app.post('/telemetry', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("control_signal", data)
        self.assertIn("ai_insights", data)
        self.assertIn("recon_loss", data["ai_insights"])

    def test_receiver_live_status_endpoint(self):
        resp = self.app.get('/api/live-status')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("latest", data)
        self.assertIn("history", data)

    def test_receiver_dashboard_endpoint(self):
        resp = self.app.get('/dashboard')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Multi-Modal AI IoT Controller", resp.data)

    def test_threshold_api_endpoints(self):
        # 1. Test GET /api/thresholds
        resp = self.app.get('/api/thresholds')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("thresholds", data)
        self.assertIn("temp_trip_threshold", data["thresholds"])

        # 2. Test POST /api/set-thresholds
        new_thresh = {
            "temp_trip": 34.5,
            "ai_cooling": 31.5,
            "light_min": 42.0
        }
        post_resp = self.app.post('/api/set-thresholds', json=new_thresh)
        self.assertEqual(post_resp.status_code, 200)
        res_data = post_resp.get_json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["thresholds"]["temp_trip_threshold"], 34.5)
        self.assertEqual(res_data["thresholds"]["ai_cooling_threshold"], 31.5)
        self.assertEqual(res_data["thresholds"]["light_min_threshold"], 42.0)

    def test_target_setpoint_endpoints(self):
        # Test GET /api/target
        resp = self.app.get('/api/target')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("target", data)
        self.assertIn("target_temp", data["target"])

        # Test POST /api/set-target
        new_target = {
            "target_temp": 26.8,
            "target_light": 55.0
        }
        post_resp = self.app.post('/api/set-target', json=new_target)
        self.assertEqual(post_resp.status_code, 200)
        res_data = post_resp.get_json()
        self.assertEqual(res_data["status"], "success")
        self.assertEqual(res_data["target"]["target_temp"], 26.8)
        self.assertEqual(res_data["target"]["target_light"], 55.0)

    def test_ai_target_regulation(self):
        self.engine.update_target(target_temp=27.0, target_light=50.0)
        # Test temperature at target setpoint (27.0°C)
        res = self.engine.process_telemetry(27.0, 60.0)
        self.assertTrue(res["target_achieved"])
        self.assertEqual(res["target_temp"], 27.0)
        self.assertEqual(res["temp_error"], 0.0)
        self.assertIn("TARGET", res["recommendation"])

    def test_arduino_bridge_multimodal_parsers(self):
        # JSON format
        j = parse_sensor_line('{"temperature": 28.4, "light": 62.0}')
        self.assertIsNotNone(j)
        self.assertEqual(j["temperature"], 28.4)
        self.assertEqual(j["light"], 62.0)

        # Comma-separated
        c = parse_sensor_line('27.5, 45.0')
        self.assertIsNotNone(c)
        self.assertEqual(c["temperature"], 27.5)
        self.assertEqual(c["light"], 45.0)

        # Key-value
        kv = parse_sensor_line('temp=29.2, light=80.5')
        self.assertIsNotNone(kv)
        self.assertEqual(kv["temperature"], 29.2)
        self.assertEqual(kv["light"], 80.5)

if __name__ == '__main__':
    unittest.main()
