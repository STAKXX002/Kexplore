/*
 * =====================================================================
 * Project: Multi-Modal AI IoT Controller (Temperature + Light/LDR)
 * Target: Arduino Uno / Nano / Mega
 * =====================================================================
 * Features:
 * - Reads 2 Heterogeneous Sensors:
 *     1. Temperature Sensor: Analog Pin A0 (LM35 / TMP36 or Potentiometer: 20-45 °C)
 *     2. Light / LDR Sensor: Analog Pin A1 (Photoresistor: 0-100% Illumination)
 * - Controls 2 Distinct Actuators:
 *     1. Actuator LED 1: Digital Pin 8 (Thermal Actuator / Cooling Fan Relay)
 *     2. Actuator LED 2: Digital Pin 9 (Smart Illumination / Daylight Harvesting)
 * - Streams JSON telemetry periodically over Serial:
 *     {"temperature": 28.5, "light": 65.2}
 * - Receives real-time AI actuation commands:
 *     {"led_1": 1, "led_2": 0}
 * =====================================================================
 */

const int PIN_TEMP_SENSOR = A0;   // Sensor 1: Temperature
const int PIN_LDR_SENSOR  = A1;   // Sensor 2: LDR Light Sensor
const int PIN_LED_COOLING = 8;    // Actuator 1: Cooling / Thermal Relay
const int PIN_LED_LIGHT   = 9;    // Actuator 2: Smart Illumination

// Telemetry interval (2000 ms = 2 seconds)
const unsigned long TELEMETRY_INTERVAL = 2000;
unsigned long lastTelemetryTime = 0;

String inputBuffer = "";

// Helper to read Temperature (e.g., LM35: 10mV/°C or linear mapping 20-45°C)
float readTemperature() {
  int raw = analogRead(PIN_TEMP_SENSOR);
  // Linear scaling: raw 0-1023 maps to 20.0°C - 45.0°C
  float temp = 20.0 + ((float)raw / 1023.0) * 25.0;
  return round(temp * 10.0) / 10.0;
}

// Helper to read Light / LDR percentage (0.0% to 100.0%)
float readLightLevel() {
  int raw = analogRead(PIN_LDR_SENSOR);
  // 0 -> 0% (Darkness), 1023 -> 100% (Full Light)
  float light = ((float)raw / 1023.0) * 100.0;
  return round(light * 10.0) / 10.0;
}

void setup() {
  Serial.begin(9600);
  
  pinMode(PIN_LED_COOLING, OUTPUT);
  pinMode(PIN_LED_LIGHT, OUTPUT);
  
  digitalWrite(PIN_LED_COOLING, LOW);
  digitalWrite(PIN_LED_LIGHT, LOW);
  
  // Power-on sequence: quick LED flashes
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED_COOLING, HIGH);
    digitalWrite(PIN_LED_LIGHT, HIGH);
    delay(120);
    digitalWrite(PIN_LED_COOLING, LOW);
    digitalWrite(PIN_LED_LIGHT, LOW);
    delay(120);
  }
}

void loop() {
  unsigned long now = millis();

  // 1. Process incoming AI commands from Python bridge
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

  // 2. Periodically send telemetry payload as JSON
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = now;
    
    float current_temp = readTemperature();
    float current_light = readLightLevel();

    // Send JSON telemetry
    Serial.print("{\"temperature\":");
    Serial.print(current_temp, 1);
    Serial.print(",\"light\":");
    Serial.print(current_light, 1);
    Serial.println("}");
  }
}

void processCommand(String cmd) {
  cmd.trim();
  
  // Parse LED 1 (Cooling / Thermal Load)
  if (cmd.indexOf("\"led_1\":1") >= 0 || cmd.indexOf("\"led_1\": 1") >= 0 || cmd.indexOf("LED1=1") >= 0) {
    digitalWrite(PIN_LED_COOLING, HIGH);
  } else if (cmd.indexOf("\"led_1\":0") >= 0 || cmd.indexOf("\"led_1\": 0") >= 0 || cmd.indexOf("LED1=0") >= 0) {
    digitalWrite(PIN_LED_COOLING, LOW);
  }

  // Parse LED 2 (Smart Illumination)
  if (cmd.indexOf("\"led_2\":1") >= 0 || cmd.indexOf("\"led_2\": 1") >= 0 || cmd.indexOf("LED2=1") >= 0) {
    digitalWrite(PIN_LED_LIGHT, HIGH);
  } else if (cmd.indexOf("\"led_2\":0") >= 0 || cmd.indexOf("\"led_2\": 0") >= 0 || cmd.indexOf("LED2=0") >= 0) {
    digitalWrite(PIN_LED_LIGHT, LOW);
  }
}
