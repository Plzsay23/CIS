import serial
import json
import websocket
import time

# ====================================================================
# ⚙️ C.I.S Edge-Hub 네트워크 및 하드웨어 환경 설정
# ====================================================================
LAPTOP_IP = "100.90.245.102"
TOMCAT_PORT = "8081"
PROJECT_NAME = "CIS_Server"

WS_URL = f"ws://{LAPTOP_IP}:{TOMCAT_PORT}/{PROJECT_NAME}/eggstream"

SERIAL_PORT = "/dev/arduino"
BAUD_RATE = 9600

RECONNECT_DELAY_SEC = 5
SERIAL_READ_DELAY_SEC = 0.05


def on_message(ws, message):
    # 서버에서 내려오는 메시지가 있으면 여기서 처리
    pass


def on_error(ws, error):
    msg = str(error)
    print(f"⚠️ [웹소켓 에러]: {msg}")

    if "Handshake status 404" in msg:
        print("💡 404입니다. IP/방화벽 문제가 아니라 Tomcat Context Root 또는 @ServerEndpoint 경로 문제입니다.")
        print(f"   현재 요청 URL: {WS_URL}")
    elif "Connection refused" in msg:
        print("💡 Tomcat이 해당 포트에서 열려 있지 않습니다. 포트/서버 실행 상태를 확인하세요.")
    elif "timed out" in msg:
        print("💡 네트워크 응답 시간 초과입니다. IP, 라우팅, Tailscale/LAN 연결을 확인하세요.")
    else:
        print("💡 Eclipse Console의 Tomcat/WebSocket 로그를 확인하세요.")


def on_close(ws, close_status_code, close_msg):
    print(f"❌ [관제망 연결 해제] code={close_status_code}, msg={close_msg}")


def build_payload_from_arduino_json(sensor):
    """
    arduino.ino 출력 형식:
    {
      "uptime_ms": 12345,
      "co2_ppm": 430 또는 null,
      "co2_error": null 또는 -1/-2/-3,
      "temperature_c": 25.4 또는 null,
      "humidity_percent": 48.0 또는 null,
      "dht_error": false/true,
      "nh3_raw": 70,
      "nh3_voltage": 0.342,
      "nh3_unit": "raw_adc"
    }

    Java 서버는 현재 "co2", "temp", "humid" 키를 문자열 파싱하므로
    여기서 기존 서버 규격으로 다시 매핑한다.
    """
    co2 = sensor.get("co2_ppm")
    temp = sensor.get("temperature_c")
    humid = sensor.get("humidity_percent")
    nh3_raw = sensor.get("nh3_raw")
    nh3_voltage = sensor.get("nh3_voltage")

    # Java 쪽 코드가 null을 parseDouble 할 수 없으므로 핵심 값이 없으면 전송하지 않음.
    if co2 is None:
        raise ValueError(f"CO2 값 없음: co2_error={sensor.get('co2_error')}")
    if temp is None or humid is None:
        raise ValueError(f"DHT 값 없음: dht_error={sensor.get('dht_error')}")
    if nh3_raw is None or nh3_voltage is None:
        raise ValueError("NH3 값 없음")

    payload = {
        "type": "robot_update",

        # 기존 Java / Dashboard 호환 키
        "co2": int(co2),
        "temp": float(temp),
        "humid": float(humid),

        # 기존 dashboard가 nh3 하나만 볼 경우를 위해 전압값을 대표값으로 사용
        "nh3": float(nh3_voltage),

        # 새 Arduino 원본값도 같이 보존
        "uptime_ms": int(sensor.get("uptime_ms", 0)),
        "co2_error": sensor.get("co2_error"),
        "dht_error": bool(sensor.get("dht_error", False)),
        "nh3_raw": int(nh3_raw),
        "nh3_voltage": float(nh3_voltage),
        "nh3_unit": sensor.get("nh3_unit", "raw_adc"),

        "side": "left",
    }

    return payload


def on_open(ws):
    print("🔗 [Jetson ➡️ Laptop Eclipse] 무선 웹소켓 파이프라인 개방 성공")

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            # Arduino UNO는 시리얼 포트를 열면 리셋될 수 있으므로 잠깐 대기
            time.sleep(2)
            ser.reset_input_buffer()

            print(f"🤖 아두이노({SERIAL_PORT}) 시리얼 포트 개방 완료. JSON 스트리밍을 시작합니다.\n")

            while getattr(ws, "keep_running", True):
                if ser.in_waiting <= 0:
                    time.sleep(SERIAL_READ_DELAY_SEC)
                    continue

                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                try:
                    sensor = json.loads(line)
                    payload = build_payload_from_arduino_json(sensor)

                    ws.send(json.dumps(payload, ensure_ascii=False))

                    print(
                        "🚀 [환경 센서 데이터 전송] "
                        f"CO2={payload['co2']}ppm | "
                        f"Temp={payload['temp']}°C | "
                        f"Humid={payload['humid']}% | "
                        f"NH3(raw)={payload['nh3_raw']} | "
                        f"NH3(V)={payload['nh3_voltage']}"
                    )

                except json.JSONDecodeError:
                    print(f"⚠️ [JSON 파싱 실패] 아두이노 원문: {line}")
                except ValueError as e:
                    print(f"⚠️ [센서값 스킵] {e} | 원문: {line}")
                except websocket.WebSocketConnectionClosedException:
                    print("❌ [웹소켓 전송 실패] 연결이 이미 닫혔습니다.")
                    break

                time.sleep(SERIAL_READ_DELAY_SEC)

    except serial.SerialException as e:
        print(f"❌ [하드웨어 연결 오류] 아두이노 포트({SERIAL_PORT})를 열 수 없습니다: {e}")
        ws.close()


def connect_and_run():
    print(f"📡 [C.I.S 원격 브릿지] {WS_URL} 연결 시도 중...")

    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.on_open = on_open
    ws.run_forever()


if __name__ == "__main__":
    while True:
        try:
            connect_and_run()
        except KeyboardInterrupt:
            print("\n사용자 종료")
            break
        except Exception as e:
            print(f"❌ [브릿지 오류]: {e}")

        print(f"{RECONNECT_DELAY_SEC}초 후 재연결합니다...")
        time.sleep(RECONNECT_DELAY_SEC)