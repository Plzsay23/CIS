import serial
import json
import websocket
import time

# ====================================================================
# ⚙️ C.I.S Edge-Hub 네트워크 및 하드웨어 환경 설정
# ====================================================================
LAPTOP_IP = "100.90.245.102"  # 노트북 무선 LAN IPv4 주소
TOMCAT_PORT = "8081"           # 이클립스 톰캣 포트 번호
PROJECT_NAME = "CIS_Server"    # 이클립스 Dynamic Web Project 이름

# 최종 무선 웹소켓 엔드포인트 주소 매핑
WS_URL = f"ws://{LAPTOP_IP}:{TOMCAT_PORT}/{PROJECT_NAME}/eggstream"

# 아두이노 고유 식별 심볼릭 링크 경로 적용 ⚡
SERIAL_PORT = '/dev/arduino'  
BAUD_RATE = 9600

def on_message(ws, message):
    pass

def on_error(ws, error):
    print(f"⚠️ [네트워크 에러]: {error}")
    print("💡 팁: 노트북의 'Windows 방화벽'이 꺼져있는지, 이클립스 톰캣이 가동 중인지 확인하세요.")

def on_close(ws, close_status_code, close_msg):
    print("❌ [관제망 연결 해제] 5초 후 재연결 파이프라인을 가동합니다...")
    time.sleep(5)
    connect_and_run()

def on_open(ws):
    print("🔗 [Jetson ➡️ Laptop Eclipse] 무선 웹소켓 파이프라인 개방 성공!")
    
    try:
        # 아두이노 시리얼 포트 개방
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.flush() # 초기 버퍼 비우기
        print(f"🤖 아두이노({SERIAL_PORT}) 통신 개방 완료. 실시간 스트리밍을 시작합니다.\n")
        
        while True:
            if ser.in_waiting > 0:
                # 1. 아두이노가 보낸 데이터 한 줄 읽기 (이제 순수 JSON 포맷 텍스트입니다.)
                raw_line = ser.readline().decode('utf-8').rstrip()
                
                # 빈 줄이나 노이즈 예외 처리
                if not raw_line or not raw_line.startswith("{") or not raw_line.endswith("}"):
                    continue
                
                try:
                    # 2. 아두이노가 던진 JSON 텍스트를 파이썬 딕셔너리로 변환
                    arduino_data = json.loads(raw_line)
                    
                    # 3. 센서 에러 상태 조기 필터링 및 디폴트값 예외처리
                    # 만약 아두이노가 에러로 null을 보냈다면 기본값(0) 처리하여 시스템 붕괴 차단
                    co2 = arduino_data.get("co2_ppm")
                    co2_val = int(co2) if co2 is not None else 0
                    
                    temp = arduino_data.get("temperature_c")
                    temp_val = float(temp) if temp is not None else 0.0
                    
                    humid = arduino_data.get("humidity_percent")
                    humid_val = float(humid) if humid is not None else 0.0
                    
                    # NH3 전압 기반 관제용 정보 가공 (주석에 명시된 Linux 연산 역할 반영)
                    nh3_raw = arduino_data.get("nh3_raw", 0)
                    nh3_voltage = arduino_data.get("nh3_voltage", 0.0)
                    
                    # 4. 자바 웹소켓 서버가 대기하는 통합 관제 표준 규격(JSON)으로 재포장
                    payload = {
                        "type": "robot_update",
                        "co2": co2_val,
                        "temp": temp_val,
                        "humid": humid_val,
                        "nh3": nh3_voltage, # 자바 서버 테이블에 실수형태로 입력되도록 매핑
                        "side": "left"
                    }
                    
                    # 5. 무선 망을 타고 노트북 자바 서버로 패킷 전송
                    ws.send(json.dumps(payload))
                    print(f"🚀 [환경 데이터 무선 전송]: CO2={co2_val}ppm | Temp={temp_val}°C | Humid={humid_val}% | NH3_V={nh3_voltage}V")
                    
                except json.JSONDecodeError:
                    print("⚠️ [데이터 손실] 시리얼 버퍼가 깨진 불완전한 JSON 패킷을 감지하여 스킵합니다.")
                except ValueError as ve:
                    print(f"⚠️ [데이터 파싱 오차]: {ve}")
                        
            time.sleep(0.1) # CPU 가부하 방지용 미세 딜레이
            
    except serial.SerialException as e:
        print(f"❌ [하드웨어 오류] 아두이노가 {SERIAL_PORT} 경로에 없습니다: {e}")
        ws.close()

def connect_and_run():
    print(f"📡 [C.I.S 원격 브릿지] {WS_URL} 연결 시도 중...")
    ws = websocket.WebSocketApp(WS_URL,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

if __name__ == "__main__":
    connect_and_run()