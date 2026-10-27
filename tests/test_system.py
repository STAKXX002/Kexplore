"""
Automated Test Suite for Multi-Modal AI IoT Telemetry & Actuation System.
Tests ML Engine, PyTorch Autoencoder, Flask Endpoints, SCADA Analytics,
Arduino Serial Parsers, and Counterfactual MPC Setpoint Regulation.
"""

import unittest
import sys
import os

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ml.engine import MLEngine
from src.server.app import app
from src.server.analytics import SessionAnalytics
from src.hardware.bridge import parse_sensor_line


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
            "temperature": 27.2,
            "light": 55.0
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
        self.assertIn("analytics", data)

    def test_receiver_dashboard_endpoint(self):
        resp = self.app.get('/dashboard')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Multi-Modal AI IoT Controller", resp.data)
        resp.close()
        
        # Verify modular static assets
        resp_css = self.app.get('/static/css/dashboard.css')
        self.assertEqual(resp_css.status_code, 200)
        resp_css.close()
        resp_js = self.app.get('/static/js/dashboard.js')
        self.assertEqual(resp_js.status_code, 200)
        resp_js.close()

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

    def test_export_csv_endpoint(self):
        self.app.post('/telemetry', json={"temperature": 27.1, "light": 50.0})
        resp = self.app.get('/api/export-csv')
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers["Content-Type"])
        self.assertIn("attachment", resp.headers["Content-Disposition"])
        self.assertTrue(resp.data.startswith(b"timestamp,temperature,light"))
        resp.close()

    def test_demo_injection_endpoint(self):
        resp = self.app.post('/api/demo/inject-state', json={"mode": "SOLAR_SURGE"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.app.get('/api/demo/current-mode').get_json()["mode"], "SOLAR_SURGE")

        # Telemetry response carries the mode so simulator can follow it
        tel = self.app.post('/telemetry', json={"temperature": 27.0, "light": 55.0}).get_json()
        self.assertEqual(tel["demo_mode"], "SOLAR_SURGE")

        bad = self.app.post('/api/demo/inject-state', json={"mode": "NOT_A_MODE"})
        self.assertEqual(bad.status_code, 400)

        self.app.post('/api/demo/inject-state', json={"mode": "NORMAL"})

    def test_audit_event_logging(self):
        analytics = SessionAnalytics()
        base = {
            "target_temp": 27.0,
            "control_mode": "AI_PREDICTIVE",
            "predicted_temp": 27.0,
            "recon_loss": 0.01,
            "anomaly_reason": "Normal",
            "led_2": 0
        }

        analytics.record_sample(26.0, 50.0, {**base, "led_1": 1, "tripped": False, "is_anomaly": False}, 3.0, 0)
        # AI turns load OFF below target because forecast reaches setpoint -> AI_MPC early cut-off
        analytics.record_sample(26.6, 50.0, {**base, "led_1": 0, "tripped": False, "is_anomaly": False,
                                             "predicted_temp": 27.1}, 3.0, 2000)
        analytics.record_sample(34.0, 5.0, {**base, "led_1": 0, "tripped": False, "is_anomaly": True,
                                            "anomaly_reason": "Dark heat"}, 3.0, 4000)

        cats = [e["category"] for e in analytics.snapshot()["events"]]
        self.assertIn("AI_MPC", cats)
        self.assertIn("AUTOENCODER", cats)
        self.assertEqual(analytics.summary()["anomaly_events"], 1)

    def test_performance_metrics(self):
        analytics = SessionAnalytics(stability_band=0.35)
        base = {
            "target_temp": 27.0,
            "control_mode": "AI_PREDICTIVE",
            "predicted_temp": 27.0,
            "recon_loss": 0.01,
            "anomaly_reason": "Normal",
            "led_2": 0,
            "tripped": False,
            "is_anomaly": False
        }
        # 4 samples below target: thermostat would be ON for all 4, AI is ON for 2 -> 50% saved
        for i, (t, led) in enumerate([(26.0, 1), (26.3, 1), (26.6, 0), (26.8, 0)]):
            analytics.record_sample(t, 50.0, {**base, "led_1": led}, 2.0, i * 60000)

        perf = analytics.snapshot()["performance"]
        self.assertEqual(perf["duty_saved_pct"], 50.0)
        self.assertEqual(perf["ai_duty_pct"], 50.0)
        self.assertEqual(perf["stability_pct"], 25.0)  # only 26.8 is within +-0.35
        self.assertEqual(perf["latency_ms"], 2.0)
        # 26.0 -> 26.8 over 3 minutes
        self.assertAlmostEqual(perf["momentum_c_per_min"], 0.27, places=2)

    def test_live_status_includes_analytics(self):
        self.app.post('/telemetry', json={"temperature": 27.0, "light": 55.0, "source": "simulator"})
        data = self.app.get('/api/live-status').get_json()
        a = data["analytics"]
        for key in ["performance", "heartbeat", "sensor_health", "events", "summary", "demo_mode"]:
            self.assertIn(key, a)
        self.assertEqual(a["heartbeat"]["source"], "simulator")
        self.assertEqual(a["heartbeat"]["status"], "LIVE")

    def test_ai_cold_start_and_mpc_coasting_vs_classical(self):
        # 1. Cold start at ~24.7C (below 27.0C target): XGBoost must NOT hallucinate ~29.5C,
        #    and AI must turn LED 1 ON to heat toward the 27.0C setpoint.
        cold_engine = MLEngine()
        cold_engine.update_target(target_temp=27.0, target_light=50.0)
        for t in [25.2, 25.0, 24.8, 24.7, 24.7]:
            out = cold_engine.process_telemetry(t, 55.0)
        self.assertEqual(out["led_1"], 1)
        self.assertLess(out["predicted_temp"], 26.8)
        self.assertGreater(out["predicted_temp"], 23.5)

        # 2. Closed-loop 60-step comparison under identical second-order thermal inertia
        def run_closed_loop(mode):
            eng = MLEngine()
            eng.set_control_mode(mode)
            eng.update_target(target_temp=27.0, target_light=50.0)
            temp, vel, led_1 = 25.2, 0.0, 0
            peak = temp
            steady_in_band = 0
            for step in range(60):
                res = eng.process_telemetry(round(temp, 2), 55.0)
                led_1 = res["led_1"]
                pull = -0.030 * (temp - 23.0)
                drive = 0.40 if led_1 == 1 else 0.0
                vel = 0.78 * vel + 0.22 * (drive + pull + 0.004)
                temp += vel
                peak = max(peak, temp)
                if step >= 20 and abs(temp - 27.0) <= 0.35:
                    steady_in_band += 1
            return peak, steady_in_band

        ai_peak, ai_in_band = run_closed_loop("AI_PREDICTIVE")
        cl_peak, cl_in_band = run_closed_loop("CLASSICAL")
        self.assertLess(ai_peak, cl_peak)
        self.assertGreaterEqual(ai_in_band, 35)

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
