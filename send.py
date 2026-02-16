import json
import time
import cv2
import numpy as np
import paho.mqtt.client as mqtt

# ---------------- MQTT ----------------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_golf/track"

# ---------------- KEY MAPS ----------------
# เลือก "สี" ด้วยตัวเลข
# (ปรับ mapping ได้ตามใจ)
COLOR_BY_NUM = {
    ord('1'): "orange",
    ord('2'): "red1",
    ord('3'): "red2",
    ord('4'): "pink",
    ord('5'): "light_green",
}

# เลือก "คลาส" ด้วย qwertyuiop...
# (ปรับ mapping ได้ตามใจ)
CLASS_BY_QWERTY = {
    ord('q'): "bridgestone",
    ord('w'): "callaway",
    ord('e'): "mizuno",
    ord('r'): "nike",
    ord('t'): "titleist",
    ord('y'): "unknown",   # เผื่อไว้: เลือก unknown แบบเร็ว
    # ord('u'): "...",
    # ord('i'): "...",
    # ord('o'): "...",
    # ord('p'): "...",
}

CONF_STEP = 0.05
CONF_MIN = 0.00
CONF_MAX = 0.99

WIN_NAME = "MQTT Test Sender (PC)"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def publish_mqtt(client, color_name, class_name, conf):
    payload = {
        "color": color_name,
        "class": class_name,
        "confidence": None if conf is None else round(float(conf), 2),
    }
    client.publish(MQTT_TOPIC, json.dumps(payload))
    return payload

def draw_right_list(canvas, title, items, x_right, y_start, font, scale, thickness, line_gap):
    # วัด max width เพื่อชิดขวา
    sizes = [cv2.getTextSize(s, font, scale, thickness)[0] for s in items]
    max_w = max((w for w, h in sizes), default=0)

    cv2.putText(canvas, title, (x_right - max_w, y_start), font, scale + 0.1, (255, 255, 255), thickness+1)
    y = y_start + 35
    for s in items:
        (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
        x = x_right - max_w
        cv2.putText(canvas, s, (x, y), font, scale, (220, 220, 220), thickness)
        y += th + line_gap
    return y

def draw_ui(canvas, color_name, class_name, conf, last_sent):
    h, w = canvas.shape[:2]
    canvas[:] = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(canvas, "MQTT Keyboard Sender (PC)", (30, 55), font, 1.2, (255, 255, 255), 2)

    # Current selection
    conf_text = "None" if conf is None else f"{conf:.2f}"
    cv2.putText(canvas, f"COLOR      : {color_name}", (30, 120), font, 0.95, (0, 255, 255), 2)
    cv2.putText(canvas, f"CLASS      : {class_name}", (30, 165), font, 0.95, (0, 255, 0), 2)
    cv2.putText(canvas, f"CONFIDENCE : {conf_text}", (30, 210), font, 0.95, (255, 255, 255), 2)

    # Controls (left-center)
    help_lines = [
        "Send    = S or Enter",
        "Conf    = +/-   |  Reset conf = R",
        "Quit    = ESC",
    ]
    x_help = 30  # ระยะจากขอบซ้าย
    bottom_margin = 70  # ระยะจากขอบล่าง
    line_step = 28  # ระยะห่างแต่ละบรรทัด

    # คำนวณให้ชุดข้อความจบที่ขอบล่าง
    y = h - bottom_margin - (len(help_lines) - 1) * line_step

    for line in help_lines:
        cv2.putText(canvas, line, (x_help, y), font, 0.7, (200, 200, 200), 2)
        y += line_step

    # Right side lists (vertical)
    right_x = w - 30
    color_items = [f"{chr(k)} : {v}" for k, v in sorted(COLOR_BY_NUM.items(), key=lambda x: x[0])]
    class_items = [f"{chr(k)} : {v}" for k, v in sorted(CLASS_BY_QWERTY.items(), key=lambda x: x[0])]

    y2 = 110
    y2 = draw_right_list(canvas, "COLOR :", color_items, right_x, y2, font, 0.75, 2, 10)
    y2 += 20
    draw_right_list(canvas, "CLASS :", class_items, right_x, y2, font, 0.75, 2, 10)

    # Last sent
    if last_sent is not None:
        cv2.putText(canvas, f"LAST SENT: {last_sent}", (30, h - 30), font, 0.45, (0, 200, 255), 2)

def main():
    # MQTT connect
    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # default values
    color_name = list(COLOR_BY_NUM.values())[0] if COLOR_BY_NUM else "orange"
    class_name = "unknown"
    conf = None
    last_sent = None

    # Bigger window (not fullscreen)
    canvas = np.zeros((540, 760, 3), dtype=np.uint8)  # ✅ ใหญ่ขึ้น อ่านง่าย
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 760, 540)

    try:
        while True:
            draw_ui(canvas, color_name, class_name, conf, last_sent)
            cv2.imshow(WIN_NAME, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue

            # Quit
            if key == 27:
                break

            # Select color by numbers
            if key in COLOR_BY_NUM:
                color_name = COLOR_BY_NUM[key]
                continue

            # Select class by qwerty keys
            if key in CLASS_BY_QWERTY:
                class_name = CLASS_BY_QWERTY[key]
                # ถ้าเลือก unknown ให้ conf เป็น None
                if str(class_name).lower() == "unknown":
                    conf = None
                else:
                    if conf is None:
                        conf = 0.50
                continue

            # Confidence up/down
            if key in (ord('+'), ord('='), ord(']')):
                if conf is None:
                    conf = 0.30
                conf = clamp(conf + CONF_STEP, CONF_MIN, CONF_MAX)
                continue

            if key in (ord('-'), ord('_'), ord('[')):
                if conf is None:
                    conf = 0.30
                conf = clamp(conf - CONF_STEP, CONF_MIN, CONF_MAX)
                continue

            # reset conf
            if key in (ord('r'), ord('R')):
                conf = None if str(class_name).lower() == "unknown" else 0.30
                continue

            # Send (S or Enter)
            if key in (ord('s'), ord('S'), 13):
                conf_to_send = None if str(class_name).lower() == "unknown" else (0.30 if conf is None else conf)
                last_sent = publish_mqtt(client, color_name, class_name, conf_to_send)
                print("Published:", last_sent)
                continue

    finally:
        client.loop_stop()
        client.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
