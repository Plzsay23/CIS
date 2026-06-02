#include <SoftwareSerial.h>
#include <DHT.h>

/*
  Arduino environmental sensor raw JSON sender

  Sensors:
    - MH-Z19B CO2 sensor via SoftwareSerial
    - DHT11 temperature/humidity sensor
    - DFRobot SEN0567 NH3 sensor analog output

  Arduino only sends:
    - sensor values
    - sensor read errors

  Linux / Jetson should calculate:
    - risk level
    - ventilation status
    - filtering
    - DB insert
    - dashboard status

  Wiring:

  MH-Z19B:
    TX  -> Arduino D2
    RX  -> Arduino D3
    VCC -> Arduino 5V
    GND -> Arduino GND

  DHT11:
    DATA -> Arduino D4
    VCC  -> Arduino 5V
    GND  -> Arduino GND

  SEN0567:
    A / SIG / AOUT -> Arduino A0
    VCC            -> Arduino 5V
    GND            -> Arduino GND

  SEN0567 pulldown:
    Arduino A0 -> 10kΩ -> Arduino GND
*/

// -------------------- Pin config --------------------
#define MHZ_RX_PIN 2
#define MHZ_TX_PIN 3

#define DHTPIN 4
#define DHTTYPE DHT11

#define NH3_PIN A0

// -------------------- Sensor objects --------------------
SoftwareSerial mhz19(MHZ_RX_PIN, MHZ_TX_PIN);
DHT dht(DHTPIN, DHTTYPE);

// -------------------- ADC config --------------------
const float ADC_REF_VOLTAGE = 5.0;
const float ADC_MAX_VALUE = 1023.0;

// -------------------- Timing --------------------
const unsigned long SENSOR_INTERVAL_MS = 5000;
unsigned long lastReadTime = 0;

// -------------------- MH-Z19B command --------------------
byte readCO2Command[9] = {
  0xFF, 0x01, 0x86,
  0x00, 0x00, 0x00,
  0x00, 0x00, 0x79
};

byte mhzResponse[9];

// Debug flag.
// true로 바꾸면 MH-Z19B UART raw bytes를 출력함.
// Python JSON 파서와 같이 쓸 때는 반드시 false 유지.
const bool PRINT_MHZ_RAW_DEBUG = false;

// -------------------- Utility --------------------
byte getChecksum(byte *packet) {
  byte checksum = 0;

  for (int i = 1; i < 8; i++) {
    checksum += packet[i];
  }

  checksum = 0xFF - checksum;
  checksum += 1;

  return checksum;
}

float rawToVoltage(int raw) {
  return raw * (ADC_REF_VOLTAGE / ADC_MAX_VALUE);
}

// -------------------- SEN0567 NH3 read --------------------
// ppm이 아니라 Arduino ADC raw 값이다.
// 10kΩ 풀다운을 A0-GND에 넣고 사용.
int readNH3RawStable() {
  // ADC 안정화용 더미 리드
  analogRead(NH3_PIN);
  delay(5);

  long sum = 0;
  const int samples = 10;

  for (int i = 0; i < samples; i++) {
    sum += analogRead(NH3_PIN);
    delay(3);
  }

  return sum / samples;
}

// -------------------- MH-Z19B read --------------------
// Return:
//   > 0 : CO2 ppm
//   -1  : no response / timeout
//   -2  : invalid response header
//   -3  : checksum error
int readCO2Once() {
  // Clear old bytes
  while (mhz19.available()) {
    mhz19.read();
  }

  // Send command
  mhz19.write(readCO2Command, 9);
  mhz19.flush();

  unsigned long start = millis();
  int index = 0;

  // Wait up to 1 second for 9-byte response
  while (millis() - start < 1000) {
    if (mhz19.available()) {
      mhzResponse[index++] = mhz19.read();

      if (index == 9) {
        break;
      }
    }
  }

  if (index != 9) {
    return -1;
  }

  if (PRINT_MHZ_RAW_DEBUG) {
    Serial.print("MHZ_RAW:");
    for (int i = 0; i < 9; i++) {
      if (mhzResponse[i] < 16) {
        Serial.print("0");
      }
      Serial.print(mhzResponse[i], HEX);
      Serial.print(" ");
    }
    Serial.println();
  }

  if (mhzResponse[0] != 0xFF || mhzResponse[1] != 0x86) {
    return -2;
  }

  byte checksum = getChecksum(mhzResponse);
  if (mhzResponse[8] != checksum) {
    return -3;
  }

  int ppm = mhzResponse[2] * 256 + mhzResponse[3];
  return ppm;
}

// 3회 재시도.
// 성공하면 co2_error = 0.
// 전부 실패하면 마지막 에러 코드 반환.
int readCO2Robust(int *co2Error) {
  int result = -1;
  *co2Error = -1;

  for (int i = 0; i < 3; i++) {
    result = readCO2Once();

    if (result > 0) {
      *co2Error = 0;
      return result;
    }

    *co2Error = result;
    delay(80);
  }

  return result;
}

// -------------------- setup --------------------
void setup() {
  Serial.begin(9600);
  mhz19.begin(9600);
  dht.begin();

  pinMode(NH3_PIN, INPUT);

  // Linux/Python에서 JSON만 안정적으로 받기 위해 안내문은 출력하지 않음.
}

// -------------------- loop --------------------
void loop() {
  unsigned long now = millis();

  if (now - lastReadTime < SENSOR_INTERVAL_MS) {
    return;
  }

  lastReadTime = now;

  // -------------------- CO2 --------------------
  int co2Error = 0;
  int co2Value = readCO2Robust(&co2Error);
  bool co2Ok = co2Value > 0 && co2Error == 0;

  // -------------------- DHT11 --------------------
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  bool dhtOk = !(isnan(humidity) || isnan(temperature));

  // -------------------- SEN0567 NH3 --------------------
  int nh3Raw = readNH3RawStable();
  float nh3Voltage = rawToVoltage(nh3Raw);

  // -------------------- JSON output --------------------
  Serial.print("{");

  Serial.print("\"uptime_ms\":");
  Serial.print(now);

  Serial.print(",\"co2_ppm\":");
  if (co2Ok) {
    Serial.print(co2Value);
  } else {
    Serial.print("null");
  }

  Serial.print(",\"co2_error\":");
  if (co2Ok) {
    Serial.print("null");
  } else {
    Serial.print(co2Error);
  }

  Serial.print(",\"temperature_c\":");
  if (dhtOk) {
    Serial.print(temperature, 1);
  } else {
    Serial.print("null");
  }

  Serial.print(",\"humidity_percent\":");
  if (dhtOk) {
    Serial.print(humidity, 1);
  } else {
    Serial.print("null");
  }

  Serial.print(",\"dht_error\":");
  Serial.print(dhtOk ? "false" : "true");

  Serial.print(",\"nh3_raw\":");
  Serial.print(nh3Raw);

  Serial.print(",\"nh3_voltage\":");
  Serial.print(nh3Voltage, 3);

  Serial.print(",\"nh3_unit\":\"raw_adc\"");

  Serial.print("}");
  Serial.println();
}