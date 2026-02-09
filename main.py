import cv2
import numpy as np
from ultralytics import YOLO
import json
import paho.mqtt.client as mqtt
import time

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "ai_golf/track"

# =========================
# CONFIG
# =========================
STREAM_URL = "http://10.154.239.75:5000/video_feed"
MODEL_PATH = "my_model.onnx"
CONF_THRES = 0.4
BOX_SCALE = 1.2
YOLO_INTERVAL = 10
DEVICE = "cpu"
last_sent = 0
SEND_INTERVAL = 1.0  # วินาที

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(STREAM_URL)

# =========================
# TRACKING STATE
# =========================
track_window = None
roi_hist = None
tracked_class = None
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
frame_count = 0

# =========================
# RED MASK
# =========================
def red_mask(hsv):
    mask1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    return mask1 + mask2
# =========================
# MQTT
# =========================
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_count += 1
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # =========================
    # GREEN MONITOR ZONE (center-bottom)
    # =========================
    gw, gh = 200, 200
    gx = (w - gw) // 2
    gy = h - gh - 20
    GREEN_BOX = (gx, gy, gw, gh)

    # =========================
    # YOLO DETECT (only to start tracking)
    # =========================
    if track_window is None or frame_count % YOLO_INTERVAL == 0:
        results = model.predict(
            frame,
            conf=CONF_THRES,
            device=DEVICE,
            verbose=False
        )[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for box, cls_id in zip(boxes, classes):
                cls_name = CLASS_NAMES[int(cls_id)]

                x1, y1, x2, y2 = map(int, box)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = red_mask(hsv_roi)
                red_ratio = np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

                if red_ratio < 0.25:
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bw = int((x2 - x1) * BOX_SCALE)
                bh = int((y2 - y1) * BOX_SCALE)

                sx = max(0, cx - bw // 2)
                sy = max(0, cy - bh // 2)

                track_window = (sx, sy, bw, bh)
                tracked_class = cls_name

                roi_hsv = hsv[sy:sy+bh, sx:sx+bw]
                mask = red_mask(roi_hsv)

                roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                break

    # =========================
    # CAMSHIFT TRACKING
    # =========================
    object_center = None
    rgb_text = None

    if track_window and roi_hist is not None:
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret_cs, track_window = cv2.CamShift(dst, track_window, term_crit)

        pts = cv2.boxPoints(ret_cs)
        pts = np.intp(pts)

        # วาดกรอบ track
        cv2.polylines(frame, [pts], True, (0, 0, 255), 3)

        # แสดงชื่อคลาสบนกรอบ track
        tx, ty = pts[0]
        cv2.putText(
            frame,
            tracked_class,
            (tx, ty - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        object_center = np.mean(pts, axis=0).astype(int)

        x, y, w2, h2 = cv2.boundingRect(pts)
        roi_track = frame[y:y+h2, x:x+w2]
        if roi_track.size > 0:
            b, g, r = np.mean(roi_track.reshape(-1, 3), axis=0).astype(int)
            rgb_text = f"R:{r} G:{g} B:{b}"

    # =========================
    # DRAW GREEN BOX
    # =========================
    cv2.rectangle(
        frame,
        (gx, gy),
        (gx + gw, gy + gh),
        (0, 255, 0),
        3
    )

    # =========================
    # CHECK INSIDE GREEN BOX
    # =========================
    if object_center is not None and rgb_text is not None:
        ox, oy = object_center

        if gx < ox < gx + gw and gy < oy < gy + gh:
            payload = {
                "class": tracked_class,
                "rgb": {
                    "r": int(r),
                    "g": int(g),
                    "b": int(b)
                }
            }

            now = time.time()
            if now - last_sent > SEND_INTERVAL:
                mqtt_client.publish(
                    MQTT_TOPIC,
                    json.dumps(payload)
                )
                last_sent = now

            info = f"{tracked_class} | R:{r} G:{g} B:{b}"
            cv2.putText(
                frame,
                info,
                (w - 420, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2
            )

    cv2.imshow("Red Object Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
