import cv2
from flask import Flask, Response
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import json
import threading
import paho.mqtt.client as mqtt

# =========================
# GLOBAL STATE (latest from MQTT)
# =========================
latest_color = None
latest_class = None
latest_conf = None
latest_msg_time = 0.0
state_lock = threading.Lock()

# =========================
# GPIO CONFIG
# =========================
BUTTON_PIN = 17

# (ตัวอย่าง output pins - ปรับตามวงจรจริงของคุณ)
OUT1_PIN = 23
OUT2_PIN = 24
OUT3_PIN = 25

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.setup(OUT1_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(OUT2_PIN, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(OUT3_PIN, GPIO.OUT, initial=GPIO.LOW)

# =========================
# OUTPUT CONTROL (เตรียม if/else ไว้สั่ง output)
# =========================
def handle_output(color_name, class_name, conf):
    """
    จุดเดียวสำหรับสั่ง output ตามข้อมูลที่รับจาก MQTT
    คุณใส่เงื่อนไขได้เต็มที่ใน if/elif ด้านล่าง
    """

    # ---- Example: reset all outputs first (optional) ----
    GPIO.output(OUT1_PIN, GPIO.LOW)
    GPIO.output(OUT2_PIN, GPIO.LOW)
    GPIO.output(OUT3_PIN, GPIO.LOW)

    # ---- Guard: ถ้าไม่มี class หรือเป็น unknown ----
    if class_name is None or str(class_name).lower() == "unknown":
        # ตัวอย่าง: เปิดไฟสถานะ OUT3 เมื่อ unknown
        GPIO.output(OUT3_PIN, GPIO.HIGH)
        return

    # ---- Guard: ถ้ามี confidence และต่ำเกิน (optional) ----
    if conf is not None and conf < 0.30:
        # ตัวอย่าง: ถ้า conf ต่ำ ให้ไม่ทำอะไรหรือให้ติด OUT3
        GPIO.output(OUT3_PIN, GPIO.HIGH)
        return

    # ---- Example mapping ตามชื่อคลาส (แก้ได้ตามต้องการ) ----
    # เช่น แยกแบรนด์ลูกกอล์ฟเป็น output ต่างกัน
    if class_name == "bridgestone":
        GPIO.output(OUT1_PIN, GPIO.HIGH)

    elif class_name == "callaway":
        GPIO.output(OUT2_PIN, GPIO.HIGH)

    elif class_name == "mizuno":
        GPIO.output(OUT1_PIN, GPIO.HIGH)
        GPIO.output(OUT2_PIN, GPIO.HIGH)

    elif class_name == "nike":
        GPIO.output(OUT3_PIN, GPIO.HIGH)

    elif class_name == "titleist":
        GPIO.output(OUT2_PIN, GPIO.HIGH)
        GPIO.output(OUT3_PIN, GPIO.HIGH)

    else:
        # คลาสอื่น ๆ ที่ไม่อยู่ใน list
        GPIO.output(OUT3_PIN, GPIO.HIGH)

    # ---- Example: เพิ่มเงื่อนไขตามสีด้วยก็ได้ ----
    # if color_name == "orange": ...
    # elif color_name == "pink": ...
    # เป็นต้น


# =========================
# CAMERA + FLASK
# =========================
app = Flask(__name__)
picam2 = Picamera2()

# แนะนำตั้งค่า preview/size ให้คงที่ (กัน fps แกว่ง)
# (ปรับได้ตามต้องการ)
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# =========================
# BUTTON CALLBACK (ถ่ายภาพ)
# =========================
def button_pressed(channel):
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    filename = f"capture_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print(f"✅ Saved {filename}")

GPIO.add_event_detect(
    BUTTON_PIN,
    GPIO.FALLING,
    callback=button_pressed,
    bouncetime=300
)

# =========================
# VIDEO STREAM
# =========================
STALE_SEC = 3.0  # ถ้าไม่มี MQTT ใหม่เกินกี่วินาที ให้ขึ้นว่า stale (ปรับได้)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape

        # อ่าน state แบบ thread-safe
        with state_lock:
            c = latest_color
            cls = latest_class
            conf = latest_conf
            tmsg = latest_msg_time

        # overlay
        now = time.time()
        if cls is not None or c is not None:
            stale = (now - tmsg) > STALE_SEC
            conf_text = "None" if conf is None else f"{conf:.2f}"
            cls_text = "None" if cls is None else str(cls)
            c_text = "None" if c is None else str(c)

            text = f"COLOR:{c_text} | CLASS:{cls_text} | CONF:{conf_text}"
            cv2.putText(frame, text, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if stale:
                cv2.putText(frame, "MQTT: STALE", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "MQTT: OK", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# =========================
# MQTT CONFIG (รับ color/class/confidence)
# =========================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_golf/track"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ MQTT connected")
        client.subscribe(MQTT_TOPIC)
    else:
        print("❌ MQTT failed", rc)

def on_message(client, userdata, msg):
    global latest_color, latest_class, latest_conf, latest_msg_time

    try:
        payload = json.loads(msg.payload.decode("utf-8", errors="ignore"))

        # ✅ ให้ตรงกับฝั่งส่ง: color / class / confidence
        color_name = payload.get("color", None)
        class_name = payload.get("class", None)
        conf = payload.get("confidence", None)

        # normalize confidence
        if conf is not None:
            try:
                conf = float(conf)
            except:
                conf = None

        with state_lock:
            latest_color = color_name
            latest_class = class_name
            latest_conf = conf
            latest_msg_time = time.time()

        print(f"📩 MQTT: color={latest_color}, class={latest_class}, conf={latest_conf}")

        # ✅ สั่ง output ตามข้อมูลล่าสุด
        handle_output(latest_color, latest_class, latest_conf)

    except Exception as e:
        print("❌ MQTT parse error:", e)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        GPIO.cleanup()
        client.loop_stop()
        client.disconnect()
