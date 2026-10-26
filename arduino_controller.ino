/*
 * =====================================================================
 * Project: AI/ML Dual-Sensor Telemetry & Actuation Controller
 * Target: Arduino Uno / Nano / Mega
 * =====================================================================
 * This sketch reads 2 sensors, streams telemetry over USB Serial as JSON,
 * and controls 2 actuator LEDs (or relay modules) based on control signals
 * received from the host computer running the AI controller.
 *
 * Hardware Wiring:
 * - Sensor 1: Analog Pin A0 (e.g., LM35, TMP36, LDR, or Potentiometer)
 * - Sensor 2: Analog Pin A1 (e.g., LM35, TMP36, LDR, or Potentiometer)
 * - Actuator LED 1 (Load / Cooling Actuator): Digital Pin 8 (with 220 Ohm resistor)
 * - Actuator LED 2 (Safety Trip / Warning):  Digital Pin 9 (with 220 Ohm resistor)
 * =====================================================================
 */

const int SENSOR_1_PIN = A0;
const int SENSOR_2_PIN = A1;
const int LED_1_PIN = 8;     // Actuator 1: Primary Load / Fan / Cooling
const int LED_2_PIN = 9;     // Actuator 2: Secondary Load / Trip Alarm

// Telemetry interval in milliseconds (default: 2000 ms = 2 seconds)
const unsigned long TELEMETRY_INTERVAL = 2000;
unsigned long lastTelemetryTime = 0;

String inputBuffer = "";

// Helper to convert analog reading (0-1023) to simulated temperature (20 - 45 °C)
// If using LM35: float temp = (reading * 5.0 / 1024.0) * 100.0;
// If using a test potentiometer: maps 0-1023 to 20.0-40.0 °C
float readSensorTemp(int pin) {
  int raw = analogRead(pin);
  // Linear scaling: 0 -> 20.0°C, 1023 -> 42.0°C
  float temp = 20.0 + ((float)raw / 1023.0) * 22.0;
  return round(temp * 10.0) / 10.0;
}

void setup() {
  Serial.begin(9600);
  
  pinMode(LED_1_PIN, OUTPUT);
  pinMode(LED_2_PIN, OUTPUT);
  
  // Initial state: LEDs OFF
  digitalWrite(LED_1_PIN, LOW);
  digitalWrite(LED_2_PIN, LOW);
  
  // Power-on indicator flash
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_1_PIN, HIGH);
    digitalWrite(LED_2_PIN, HIGH);
    delay(150);
    digitalWrite(LED_1_PIN, LOW);
    digitalWrite(LED_2_PIN, LOW);
    delay(150);
  }
}

void loop() {
  unsigned long now = millis();

  // 1. Process incoming commands from Python over Serial
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += c;
    }
  }

  // 2. Periodically send sensor telemetry as JSON
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = now;
    
    float s1_val = readSensorTemp(SENSOR_1_PIN);
    float s2_val = readSensorTemp(SENSOR_2_PIN);

    // Form JSON payload
    Serial.print("{\"sensor_1\":");
    Serial.print(s1_val, 1);
    Serial.print(",\"sensor_2\":");
    Serial.print(s2_val, 1);
    Serial.println("}");
  }
}

// Simple parser for command string: {"led_1": 1, "led_2": 0} or LED1=1,LED2=0
void processCommand(String cmd) {
  cmd.trim();
  
  // Parse led_1 / LED1
  if (cmd.indexOf("\"led_1\":1") >= 0 || cmd.indexOf("\"led_1\": 1") >= 0 || cmd.indexOf("LED1=1") >= 0) {
    digitalWrite(LED_1_PIN, HIGH);
  } else if (cmd.indexOf("\"led_1\":0") >= 0 || cmd.indexOf("\"led_1\": 0") >= 0 || cmd.indexOf("LED1=0") >= 0) {
    digitalWrite(LED_1_PIN, LOW);
  }

  // Parse led_2 / LED2
  if (cmd.indexOf("\"led_2\":1") >= 0 || cmd.indexOf("\"led_2\": 1") >= 0 || cmd.indexOf("LED2=1") >= 0) {
    digitalWrite(LED_2_PIN, HIGH);
  } else if (cmd.indexOf("\"led_2\":0") >= 0 || cmd.indexOf("\"led_2\": 0") >= 0 || cmd.indexOf("LED2=0") >= 0) {
    digitalWrite(LED_2_PIN, LOW);
  }
}
