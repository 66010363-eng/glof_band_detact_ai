import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt

os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "C:\\opencv_cache"

# Video source
cap = cv2.VideoCapture("http://10.67.250.75:5000/video_feed")

# Frame / ROI config
frame_size = 640
roi_size = 320
roi_x = (frame_size - roi_size) // 2
roi_y = (frame_size - roi_size) // 2
roi_w, roi_h = roi_size, roi_size

# CHECK box (center-bottom of ROI)
check_w, check_h = 120, 60
check_x = roi_x + roi_w // 2 - check_w // 2
check_y = roi_y + roi_h - check_h - 10

# Color ranges (HSV)
color_ranges = {
    "orange": ([5, 150, 150], [15, 255, 255]),
    "red1": ([0, 150, 150], [10, 255, 255]),
    "red2": ([170, 150, 150], [180, 255, 255]),
    "pink": ([176, 99, 244], [178, 255, 255]),
    "light_green": ([31, 221, 179], [32, 255, 255])
}

# Fixed tracking box size
TRACK_W, TRACK_H = 90, 90

# YOLOv8 input size
YOLO_IN = 640

# Thresholds
CONF_ATTACH = 0.20   # ✅ ต้องมากกว่า  ถึงจะติดป้ายคลาส
CONF_DECODE = 0.03  # decode กรองเบื้องต้น (ต่ำหน่อยได้ ไม่กระทบ เพราะ attach ใช้ 0.4)
IOU_TH = 0.45

# MQTT setup
MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "ai_golf/track"
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.loop_start()

# Load ONNX model
net = cv2.dnn.readNetFromONNX("my_model.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ---- Class names (ของคุณ) ----
CLASS_NAMES = [
    "bridgestone",
    "callaway",
    "mizuno",
    "nike",
    "titleist"
]

def cls_name(cls_id):
    if cls_id is None:
        return None
    if 0 <= cls_id < len(CLASS_NAMES):
        return CLASS_NAMES[cls_id]
    return str(cls_id)

# ---------------- Helpers ----------------
def iou_xywh(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[0] + a[2], b[0] + b[2])
    yB = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = a[2] * a[3]
    areaB = b[2] * b[3]
    return inter / float(areaA + areaB - inter + 1e-5)

def clamp_box(x, y, w, h, W, H):
    x = max(0, min(x, W - w))
    y = max(0, min(y, H - h))
    return x, y, w, h

def center_in_check(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return (check_x <= cx <= check_x + check_w) and (check_y <= cy <= check_y + check_h)

def center_in_roi(cx, cy):
    return (roi_x <= cx <= roi_x + roi_w) and (roi_y <= cy <= roi_y + roi_h)

def nms_color_boxes(color_boxes, iou_th=0.25):
    """
    color_boxes item = (x,y,w,h,score, color_name)
    """
    if not color_boxes:
        return []
    boxes = sorted(color_boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if iou_xywh(cur[:4], b[:4]) < iou_th]
    return keep

def decode_yolov8_onnx(out, img_w, img_h, num_classes, conf_th=0.05):
    """
    รองรับ YOLOv8 ONNX ที่ output มักเป็น:
    - [1, (4+nc), 8400]  (ไม่มี obj)
    - [1, (5+nc), 8400]  (มี obj)
    หรือ [1, 8400, C]
    คืนค่า: (score, cls_id) เฉพาะตัวที่ score >= conf_th
    """
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = np.array(out)

    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        return []

    # ✅ ถ้าเป็น (C,N) -> transpose เป็น (N,C)
    if out.shape[0] < out.shape[1]:
        out = out.T

    N, C = out.shape
    if C < 4 + num_classes:
        return []

    # class scores (มี/ไม่มี obj)
    if C == 4 + num_classes:
        cls_scores = out[:, 4:4+num_classes]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(N), cls_ids]
    elif C == 5 + num_classes:
        obj = out[:, 4]
        cls_scores = out[:, 5:5+num_classes]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(N), cls_ids] * obj
    else:
        cls_scores = out[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(N), cls_ids]

    # เก็บเฉพาะที่ผ่าน conf_th
    keep = []
    for i in range(N):
        sc = float(scores[i])
        if sc >= conf_th:
            keep.append((sc, int(cls_ids[i])))
    return keep

def best_class_from_patch(patch):
    """
    รัน YOLO บน patch แล้วคืน (best_conf, best_cls) หรือ (None, None)
    """
    blob = cv2.dnn.blobFromImage(patch, 1/255.0, (YOLO_IN, YOLO_IN), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        out = net.forward()
    except Exception:
        return None, None

    dets = decode_yolov8_onnx(out, YOLO_IN, YOLO_IN, num_classes=len(CLASS_NAMES), conf_th=CONF_DECODE)
    if not dets:
        return None, None

    best_conf, best_cls = max(dets, key=lambda x: x[0])
    return float(best_conf), int(best_cls)

# ---------------- AI timing ----------------
last_ai_time = 0
ai_interval = 3
ai_duration = 2
ai_running = False
ai_start_time = 0

# Track state (ผูก class/conf เข้ากับกล่องสี โดย match ด้วย IoU)
tracks_prev = []  # list of dict: {"bbox":(x,y,w,h), "color":str, "class":int|None, "conf":float}

print("Enter = publish ONLY if object is inside CHECK box (prepare-send area).")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_size, frame_size))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw ROI & CHECK
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    cv2.rectangle(frame, (check_x, check_y), (check_x + check_w, check_y + check_h), (0, 0, 255), 2)
    cv2.putText(frame, "CHECK", (check_x + 8, check_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 1) Detect colors ONLY in ROI -> fixed-size boxes
    raw_fixed_boxes = []  # (x,y,w,h,score(area), color_name)

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if not center_in_roi(cx, cy):
                continue

            x = cx - TRACK_W // 2
            y = cy - TRACK_H // 2
            x, y, w, h = clamp_box(x, y, TRACK_W, TRACK_H, frame_size, frame_size)

            raw_fixed_boxes.append((x, y, w, h, float(area), color_name))

    # 2) NMS color boxes (keep color_name)
    color_boxes = nms_color_boxes(raw_fixed_boxes, iou_th=0.25)

    current_time = time.time()

    # 3) Match current color boxes to previous tracks (carry class/conf)
    tracks_curr = []
    used_prev = set()
    MATCH_IOU = 0.30

    for (x, y, w, h, score, color_name) in color_boxes:
        bbox = (x, y, w, h)

        best_i = None
        best_iou = 0.0
        for i, t in enumerate(tracks_prev):
            if i in used_prev:
                continue
            iou_v = iou_xywh(bbox, t["bbox"])
            if iou_v > best_iou:
                best_iou = iou_v
                best_i = i

        if best_i is not None and best_iou >= MATCH_IOU:
            used_prev.add(best_i)
            prev = tracks_prev[best_i]
            tracks_curr.append({
                "bbox": bbox,
                "color": color_name,
                "class": prev.get("class", None),
                "conf": float(prev.get("conf", 0.0))
            })
        else:
            tracks_curr.append({
                "bbox": bbox,
                "color": color_name,
                "class": None,
                "conf": 0.0
            })

    # 4) Run AI every ai_interval sec on each current track (classification only)
    if current_time - last_ai_time >= ai_interval:
        ai_running = True
        ai_start_time = current_time
        last_ai_time = current_time

        for t in tracks_curr:
            x, y, w, h = t["bbox"]
            patch = frame[y:y + h, x:x + w]
            if patch.size == 0:
                continue

            best_conf, best_cls = best_class_from_patch(patch)

            # ✅ ถ้าไม่มีผล หรือ conf ไม่ถึง 0.4 => ไม่ทำอะไร (ไม่โชว์ None และคงค่าที่เคยดีไว้)
            if best_conf is None or best_conf < CONF_ATTACH:
                continue

            # ✅ ถ้า conf รอบใหม่ "ต่ำกว่าของเดิม" => ไม่อัปเดต (คงเดิม)
            if best_conf <= float(t.get("conf", 0.0)):
                continue

            # ✅ อัปเดตเฉพาะตอน conf สูงกว่าเดิม
            t["class"] = best_cls
            t["conf"] = float(best_conf)

    # อัปเดต state สำหรับเฟรมถัดไป
    tracks_prev = tracks_curr

    # 5) Draw color boxes + attach class text (ONLY when class exists & conf >= 0.4)
    latest_to_send = None
    best_send_conf = -1.0

    for t in tracks_curr:
        x, y, w, h = t["bbox"]
        color_name = t["color"]

        # draw color box (always)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, color_name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # attach class text only if confident
        if t["class"] is not None and float(t.get("conf", 0.0)) >= CONF_ATTACH:
            name = cls_name(t["class"])
            conf = float(t["conf"])
            cv2.putText(frame, f"{name} ({conf:.2f})", (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # prepare sending: choose best confident track inside CHECK
        if center_in_check(x, y, w, h):
            # เลือกส่งได้ทั้งมี/ไม่มี class (ถ้าไม่มี class จะให้กรอกเองตอนกด Enter)
            conf_for_pick = float(t.get("conf", 0.0)) if t["class"] is not None else -1.0
            if conf_for_pick > best_send_conf:
                best_send_conf = conf_for_pick
                latest_to_send = t

    # AI Running text
    if ai_running:
        cv2.putText(frame, "AI Running...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if current_time - ai_start_time >= ai_duration:
            ai_running = False

    # highlight check candidate (ONLY box highlight)
    if latest_to_send is not None:
        x, y, w, h = latest_to_send["bbox"]
        if center_in_check(x, y, w, h):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            label = cls_name(latest_to_send["class"])
            if label is not None:
                cv2.putText(frame, f"TO SEND: {label}", (10, frame_size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "TO SEND: (no class)", (10, frame_size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Golf Ball Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    elif key == 13:  # Enter publish
        if latest_to_send is None:
            print("No object in CHECK to publish.")
            continue

        x0, y0, w0, h0 = latest_to_send["bbox"]
        if not center_in_check(x0, y0, w0, h0):
            print("Object moved out of CHECK. Publish blocked.")
            continue

        # ถ้ายังไม่มี class ให้กรอกเอง
        if latest_to_send["class"] is None:
            try:
                user_input = input("Enter class id (0-4) for object in CHECK box: ").strip()
                if user_input == "":
                    print("No class entered. Publish cancelled.")
                    continue
                cls_id = int(user_input)
                if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                    print("Class id out of range. Publish cancelled.")
                    continue
                latest_to_send["class"] = cls_id
                latest_to_send["conf"] = 1.0
            except Exception:
                print("Invalid input. Publish cancelled.")
                continue

        payload = {
            "color": latest_to_send["color"],
            "class": cls_name(latest_to_send["class"]),  # ✅ เป็นข้อความ
            "confidence": round(float(latest_to_send.get("conf", 0.0)), 2)
        }
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        print("Published:", payload)

cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
