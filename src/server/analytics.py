"""
Session Analytics & SCADA Audit Trail.

Tracks per-session operational metrics for the live dashboard:
- Event audit trail (state transitions + API/operator actions)
- Load duty cycle: AI controller vs a "shadow" classical thermostat
- Setpoint stability (% of recent samples inside the target band)
- AI inference latency
- Thermal momentum (dT/dt in C/min)
- Telemetry link heartbeat (source, rate, last-seen) and basic sensor health
- Presentation demo-injection state (consumed by simulator)
"""

import time
import threading
from collections import deque
from datetime import datetime

DEMO_MODES = ["NORMAL", "DARK_HEAT_SPIKE", "SOLAR_SURGE", "LDR_FAULT"]

DEMO_LABELS = {
    "NORMAL": "Normal dynamics",
    "DARK_HEAT_SPIKE": "Fire / heat hazard in darkness",
    "SOLAR_SURGE": "Solar light surge",
    "LDR_FAULT": "LDR sensor disconnect",
}


class SessionAnalytics:
    def __init__(self, stability_band=0.35, stability_window=100, event_maxlen=30):
        self.stability_band = stability_band
        self._lock = threading.Lock()
        self.events = deque(maxlen=event_maxlen)
        self.demo_mode = "NORMAL"
        self._stability_window = stability_window
        self.reset()

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #
    def reset(self):
        with self._lock:
            self.start_time = time.time()
            self.packets = 0
            self.sum_temp = 0.0
            self.sum_light = 0.0
            self.anomaly_events = 0
            self.trip_events = 0

            # Duty-cycle accounting, only for samples processed in AI mode.
            # "shadow" = what a classical bang-bang thermostat would have
            # commanded at the exact same temperature (ON while below target).
            self.ai_steps = 0
            self.ai_load_on = 0
            self.ai_shadow_on = 0
            self.classical_steps = 0
            self.classical_load_on = 0

            self.in_band = deque(maxlen=self._stability_window)
            self.latencies = deque(maxlen=20)
            self.temp_points = deque(maxlen=6)     # (timestamp_ms, temp)
            self.arrivals = deque(maxlen=10)       # server receive time (s)
            self.source = None
            self.last_temp = None
            self.last_light = None

            # Previous flags for transition detection
            self._prev = {
                "anomaly": False,
                "tripped": False,
                "locked": False,
                "led_1": None,
                "led_2": None,
            }
            self.events.clear()
        self.log_event("SYSTEM", "Session started. Awaiting telemetry...", "info")

    # ------------------------------------------------------------------ #
    # Event audit trail
    # ------------------------------------------------------------------ #
    def log_event(self, category, message, severity="info"):
        """severity: info | success | ai | warning | critical"""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "category": category,
            "message": message,
            "severity": severity,
        }
        with self._lock:
            self.events.append(entry)
        return entry

    # ------------------------------------------------------------------ #
    # Demo injection (simulator only)
    # ------------------------------------------------------------------ #
    def set_demo_mode(self, mode):
        if mode not in DEMO_MODES:
            return False
        if mode != self.demo_mode:
            self.demo_mode = mode
            sev = "info" if mode == "NORMAL" else "warning"
            self.log_event("DEMO", f"Scenario injected: {DEMO_LABELS[mode]}", sev)
        return True

    # ------------------------------------------------------------------ #
    # Per-sample recording
    # ------------------------------------------------------------------ #
    def record_sample(self, temperature, light, result, latency_ms, timestamp_ms, source=None):
        target = result["target_temp"]
        led_1 = result["led_1"]
        led_2 = result["led_2"]
        tripped = result["tripped"]
        is_anomaly = result["is_anomaly"]
        mode = result["control_mode"]
        error = temperature - target

        pending_events = []

        with self._lock:
            self.packets += 1
            self.sum_temp += temperature
            self.sum_light += light
            self.last_temp = temperature
            self.last_light = light
            if source:
                self.source = source

            if mode == "AI_PREDICTIVE":
                self.ai_steps += 1
                self.ai_load_on += led_1
                if temperature < target and not tripped and not is_anomaly:
                    self.ai_shadow_on += 1
            else:
                self.classical_steps += 1
                self.classical_load_on += led_1

            self.in_band.append(abs(error) <= self.stability_band)
            self.latencies.append(latency_ms)
            self.temp_points.append((timestamp_ms, temperature))
            self.arrivals.append(time.time())

            prev = self._prev

            # Anomaly transitions
            if is_anomaly and not prev["anomaly"]:
                self.anomaly_events += 1
                pending_events.append((
                    "AUTOENCODER",
                    f"{result['anomaly_reason']} (MSE {result['recon_loss']:.4f}). Thermal load isolated.",
                    "critical",
                ))
            elif not is_anomaly and prev["anomaly"]:
                pending_events.append(("AUTOENCODER", "Sensor pattern back to normal. Anomaly cleared.", "success"))

            # Trip transitions
            if tripped and not prev["tripped"]:
                self.trip_events += 1
                pending_events.append(("SUPERVISORY", f"Emergency trip engaged at {temperature:.1f}C.", "critical"))
            elif not tripped and prev["tripped"]:
                pending_events.append(("SUPERVISORY", f"Trip reset at {temperature:.1f}C. Resuming control.", "success"))

            # Target lock (hysteresis: lock inside band, unlock beyond 2x band)
            if not prev["locked"] and abs(error) <= self.stability_band:
                prev["locked"] = True
                pending_events.append(("TARGET", f"Locked on setpoint {target}C (actual {temperature:.2f}C).", "success"))
            elif prev["locked"] and abs(error) > 2 * self.stability_band:
                prev["locked"] = False
                pending_events.append(("TARGET", f"Left target band: {temperature:.2f}C vs {target}C.", "info"))

            # Actuator transitions
            if prev["led_1"] is not None and led_1 != prev["led_1"]:
                pred = result["predicted_temp"]
                if (led_1 == 0 and mode == "AI_PREDICTIVE" and temperature < target
                        and not is_anomaly and not tripped and pred >= target - 0.15):
                    pending_events.append((
                        "AI_MPC",
                        f"Early cut-off: XGBoost forecast {pred:.2f}C coasts to {target}C while actual is {temperature:.2f}C.",
                        "ai",
                    ))
                elif not (is_anomaly or tripped):
                    # (safety cut-offs are already explained by the AUTOENCODER / SUPERVISORY events)
                    state = "ON" if led_1 else "OFF"
                    tag = "AI_MPC" if mode == "AI_PREDICTIVE" else "CLASSICAL"
                    pending_events.append((tag, f"LED 1 (thermal load) {state} at {temperature:.2f}C.", "info"))
            if prev["led_2"] is not None and led_2 != prev["led_2"]:
                state = "ON" if led_2 else "OFF"
                pending_events.append(("LIGHTING", f"LED 2 (smart light) {state} at {light:.1f}% ambient.", "info"))

            prev["anomaly"] = is_anomaly
            prev["tripped"] = tripped
            prev["led_1"] = led_1
            prev["led_2"] = led_2

        for cat, msg, sev in pending_events:
            self.log_event(cat, msg, sev)

    # ------------------------------------------------------------------ #
    # Derived metrics
    # ------------------------------------------------------------------ #
    def _performance(self):
        ai_duty = (100.0 * self.ai_load_on / self.ai_steps) if self.ai_steps else None
        shadow_duty = (100.0 * self.ai_shadow_on / self.ai_steps) if self.ai_steps else None
        classical_duty = (100.0 * self.classical_load_on / self.classical_steps) if self.classical_steps else None

        duty_saved = None
        if self.ai_shadow_on > 0:
            duty_saved = 100.0 * (self.ai_shadow_on - self.ai_load_on) / self.ai_shadow_on

        stability = (100.0 * sum(self.in_band) / len(self.in_band)) if self.in_band else None
        latency = (sum(self.latencies) / len(self.latencies)) if self.latencies else None

        momentum = None
        if len(self.temp_points) >= 2:
            (t0, v0), (t1, v1) = self.temp_points[0], self.temp_points[-1]
            dt_min = (t1 - t0) / 60000.0
            if dt_min > 0:
                momentum = (v1 - v0) / dt_min

        def r(x, n=1):
            return round(x, n) if x is not None else None

        return {
            "duty_saved_pct": r(duty_saved),
            "ai_duty_pct": r(ai_duty),
            "shadow_duty_pct": r(shadow_duty),
            "classical_duty_pct": r(classical_duty),
            "stability_pct": r(stability),
            "stability_band": self.stability_band,
            "stability_samples": len(self.in_band),
            "latency_ms": r(latency, 2),
            "momentum_c_per_min": r(momentum, 2),
        }

    def _heartbeat(self):
        now = time.time()
        if not self.arrivals:
            return {"status": "NO_DATA", "source": None, "rate_hz": None, "last_seen_s": None}
        last_seen = now - self.arrivals[-1]
        rate = None
        if len(self.arrivals) >= 2:
            span = self.arrivals[-1] - self.arrivals[0]
            if span > 0:
                rate = (len(self.arrivals) - 1) / span
        return {
            "status": "LIVE" if last_seen < 5.0 else "STALE",
            "source": self.source or "http-client",
            "rate_hz": round(rate, 2) if rate else None,
            "last_seen_s": round(last_seen, 1),
        }

    def _sensor_health(self):
        t, l = self.last_temp, self.last_light
        if t is None:
            return {"sensor_1": "No data", "sensor_2": "No data"}
        s1 = "Healthy" if -10.0 <= t <= 80.0 else "Out of range"
        s2 = "Healthy" if 0.5 < l < 99.5 else "Saturated / disconnected?"
        return {"sensor_1": s1, "sensor_2": s2}

    def summary(self):
        with self._lock:
            runtime = time.time() - self.start_time
            perf = self._performance()
            return {
                "runtime_s": int(runtime),
                "packets": self.packets,
                "avg_temp": round(self.sum_temp / self.packets, 2) if self.packets else None,
                "avg_light": round(self.sum_light / self.packets, 1) if self.packets else None,
                "anomaly_events": self.anomaly_events,
                "trip_events": self.trip_events,
                "ai_samples": self.ai_steps,
                "classical_samples": self.classical_steps,
                **perf,
            }

    def snapshot(self):
        """Everything the dashboard needs in a single poll."""
        summary = self.summary()
        with self._lock:
            return {
                "performance": self._performance(),
                "heartbeat": self._heartbeat(),
                "sensor_health": self._sensor_health(),
                "events": list(reversed(self.events)),  # newest first
                "summary": summary,
                "demo_mode": self.demo_mode,
            }
