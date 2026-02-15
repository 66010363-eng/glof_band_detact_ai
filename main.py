import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt

os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "C:\\opencv_cache"

# ---------------- Config ----------------
cap = cv2.VideoCapture("http://10.67.250.75:5000/video_feed")

frame_size = 640
roi_size = 320
roi_x = (frame_size - roi_size) // 2
roi_y = (frame_size - roi_size) // 2
roi_w, roi_h = roi_size, roi_size

check_w, check_h = 120, 60
check_x = roi_x + roi_w // 2 - check_w // 2
check_y = roi_y + roi_h - check_h - 10

color_ranges = {
    "orange": ([5, 150, 150], [15, 255, 255]),
    "red1": ([0, 150, 150], [10, 255, 255]),
    "red2": ([170, 150, 150], [180, 255, 255]),
    "pink": ([176, 99, 244], [178, 255, 255]),
    "light_green": ([31, 221, 179], [32, 255, 255])
}

TRACK_W, TRACK_H = 90, 90
YOLO_IN = 640

# ✅ เงื่อนไขติดป้ายคลาสบนกล่องสี
CONF_ATTACH = 0.30          # ต้องมากกว่า 0.30 ถึงจะติดคลาส
CONF_DECODE = 0.05          # กรองเบื้องต้นตอน decode
MATCH_IOU = 0.30            # จับ track ต่อเฟรม

# MQTT
MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "ai_golf/track"
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.loop_start()

# YOLOv8 ONNX
net = cv2.dnn.readNetFromONNX("my_model.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

CLASS_NAMES = ["bridgestone", "callaway", "mizuno", "nike", "titleist"]

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

def center_in_roi(cx, cy):
    return (roi_x <= cx <= roi_x + roi_w) and (roi_y <= cy <= roi_y + roi_h)

def center_in_check(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return (check_x <= cx <= check_x + check_w) and (check_y <= cy <= check_y + check_h)

def nms_color_boxes(color_boxes, iou_th=0.25):
    if not color_boxes:
        return []
    boxes = sorted(color_boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        cur = boxes.pop(0)
        keep.append(cur)
        boxes = [b for b in boxes if iou_xywh(cur[:4], b[:4]) < iou_th]
    return keep

def decode_yolov8_onnx_class_only(out, num_classes, conf_th=0.05):
    """
    รองรับ YOLOv8 ONNX:
    - [1, (4+nc), 8400] (no obj)
    - [1, (5+nc), 8400] (with obj)
    - หรือ [1, 8400, C]
    คืนค่า list of (score, cls_id)
    """
    if isinstance(out, (list, tuple)):
        out = out[0]
    out = np.array(out)

    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        return []

    # (C,N) -> (N,C)
    if out.shape[0] < out.shape[1]:
        out = out.T

    N, C = out.shape
    if C < 4 + num_classes:
        return []

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

    keep = []
    for i in range(N):
        sc = float(scores[i])
        if sc >= conf_th:
            keep.append((sc, int(cls_ids[i])))
    return keep

def best_class_from_patch(patch):
    blob = cv2.dnn.blobFromImage(patch, 1/255.0, (YOLO_IN, YOLO_IN), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        out = net.forward()
    except Exception:
        return None, None

    dets = decode_yolov8_onnx_class_only(out, num_classes=len(CLASS_NAMES), conf_th=CONF_DECODE)
    if not dets:
        return None, None
    best_conf, best_cls = max(dets, key=lambda x: x[0])
    return float(best_conf), int(best_cls)

def publish_mqtt(color_name, class_text, confidence):
    payload = {
        "color": color_name,
        "class": class_text,
        "confidence": None if confidence is None else round(float(confidence), 2)
    }
    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
    print("Published:", payload)

# ---------------- Modes ----------------
MODE = "semi"  # "auto" or "semi"
AUTO_MIN_GAP = 0.6         # กัน spam
last_pub_time = 0.0
last_pub_sig = None

# ---------------- AI timing ----------------
last_ai_time = 0.0
ai_interval = 3.0
ai_running = False
ai_start_time = 0.0
ai_duration = 2.0

# Track state: ผูกคลาสกับ "กล่องสี"
# track = {"bbox":(x,y,w,h), "color":..., "class_id":..., "class_text":..., "conf":..., "area":...}
tracks_prev = []

print("Keys: m=toggle AUTO/SEMI, Enter=send (SEMI), ESC=quit")
next_track_id = 1          # id ใหม่สำหรับลูกใหม่
last_pub_track_id = None   # จำว่าล่าสุดส่งลูก id ไหนไปแล้ว

# (optional) ให้ลูกต้อง "นิ่งใน CHECK" กี่เฟรมก่อนส่ง กันเด้ง
STABLE_FRAMES = 4
stable_id = None
stable_count = 0

# (optional) ถ้าไม่มีลูกใน CHECK นานเกินนี้ จะ reset เพื่อรองรับเปลี่ยนลูก
RESET_AFTER_EMPTY_SEC = 1.0
last_seen_in_check_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_size, frame_size))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    current_time = time.time()

    # Draw ROI & CHECK
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    cv2.rectangle(frame, (check_x, check_y), (check_x + check_w, check_y + check_h), (0, 0, 255), 2)
    cv2.putText(frame, "CHECK", (check_x + 8, check_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show MODE
    cv2.putText(frame, f"MODE: {'AUTO' if MODE=='auto' else 'SEMI-AUTO'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 1) Detect colors in ROI -> fixed-size boxes + NMS
    raw_fixed_boxes = []
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

    color_boxes = nms_color_boxes(raw_fixed_boxes, iou_th=0.25)

    # 2) Match tracks (carry best class/conf forward)
    tracks_curr = []
    used_prev = set()

    for (x, y, w, h, area, color_name) in color_boxes:
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
            tid = prev.get("id", None)
            if tid is None:
                tid = 0
            tracks_curr.append({
                "id": tid,
                "bbox": bbox,
                "color": color_name,
                "area": float(area),
                "class_id": prev.get("class_id", None),
                "class_text": prev.get("class_text", None),
                "conf": float(prev.get("conf", 0.0))
            })
        else:
            tid = next_track_id
            next_track_id += 1
            tracks_curr.append({
                "id": tid,
                "bbox": bbox,
                "color": color_name,
                "area": float(area),
                "class_id": None,
                "class_text": None,
                "conf": 0.0
            })

    # 3) Run YOLO every ai_interval (classification only) + attach if conf > 0.30 and better than old
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

            # ✅ ถ้า YOLO ไม่เจอ / conf ไม่ถึง => ไม่อัปเดต (และเราไม่วาด None อยู่แล้ว)
            if best_conf is None or best_conf < CONF_ATTACH:
                continue

            # ✅ ถ้า conf ใหม่ <= เดิม => ไม่อัปเดต (กันแย่ลง)
            if best_conf <= float(t.get("conf", 0.0)):
                continue

            t["class_id"] = best_cls
            t["class_text"] = cls_name(best_cls)
            t["conf"] = float(best_conf)

    # update prev for persistence
    tracks_prev = tracks_curr

    # 4) Draw color boxes + class label (NO class None)
    latest_to_send = None
    best_pick = (-1, -1.0, -1.0)  # (has_class, conf, area)

    for t in tracks_curr:
        x, y, w, h = t["bbox"]

        # draw color box always
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, t["color"], (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # attach class only if exists and conf>=CONF_ATTACH
        if t["class_text"] is not None and float(t["conf"]) >= CONF_ATTACH:
            cv2.putText(frame, f"{t['class_text']} ({t['conf']:.2f})",
                        (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # pick candidate inside CHECK (ต้องเลือกได้แม้ไม่มีคลาส)
        if center_in_check(x, y, w, h):
            has = 1 if t["class_text"] is not None else 0
            confv = float(t["conf"]) if t["class_text"] is not None else -1.0
            areav = float(t["area"])
            score = (has, confv, areav)
            if score > best_pick:
                best_pick = score
                latest_to_send = t

    # AI Running text
    if ai_running:
        cv2.putText(frame, "AI Running...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if current_time - ai_start_time >= ai_duration:
            ai_running = False

    # highlight candidate (box only)
    if latest_to_send is not None:
        x, y, w, h = latest_to_send["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    # 5) AUTO publish: ส่งเฉพาะตอน "เปลี่ยนลูก" (track id เปลี่ยน)
    if MODE == "auto":
        if latest_to_send is None:
            # ไม่มีลูกใน CHECK -> นับเวลาเพื่อ reset (optional)
            if last_seen_in_check_time > 0 and (current_time - last_seen_in_check_time > RESET_AFTER_EMPTY_SEC):
                last_pub_track_id = None
                stable_id = None
                stable_count = 0
            # ไม่ส่งอะไร
            pass
        else:
            x, y, w, h = latest_to_send["bbox"]
            if center_in_check(x, y, w, h):
                last_seen_in_check_time = current_time

                tid = latest_to_send["id"]

                # debounce กันเด้ง (optional)
                if stable_id != tid:
                    stable_id = tid
                    stable_count = 1
                else:
                    stable_count += 1

                # ✅ เงื่อนไขส่ง: ลูกนิ่งพอ + ยังไม่เคยส่งลูก id นี้
                if stable_count >= STABLE_FRAMES and tid != last_pub_track_id:
                    class_text = latest_to_send["class_text"] if latest_to_send["class_text"] is not None else "unknown"
                    confv = latest_to_send["conf"] if latest_to_send["class_text"] is not None else None

                    publish_mqtt(latest_to_send["color"], class_text, confv)
                    last_pub_track_id = tid

    cv2.imshow("Golf Ball Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    if key in (ord('m'), ord('M')):
        MODE = "auto" if MODE == "semi" else "semi"
        print("MODE switched to:", MODE.upper())

    elif key == 13 and MODE == "semi":  # Enter only for SEMI
        if latest_to_send is None:
            print("No object in CHECK to publish.")
            continue

        x0, y0, w0, h0 = latest_to_send["bbox"]
        if not center_in_check(x0, y0, w0, h0):
            print("Object moved out of CHECK. Publish blocked.")
            continue

        # ถ้า YOLO มี class แล้ว ส่งเลย
        if latest_to_send["class_text"] is not None:
            publish_mqtt(latest_to_send["color"], latest_to_send["class_text"], latest_to_send["conf"])
            continue

        # ถ้า YOLO ไม่มี class -> ให้ user กรอก "ชื่อคลาส"
        user_input = input(f"Enter class name (e.g. {', '.join(CLASS_NAMES)}) : ").strip()
        if user_input == "":
            print("No class entered. Publish cancelled.")
            continue

        # อนุญาตพิมพ์เป็นเลขได้ด้วย (0-4)
        if user_input.isdigit():
            cls_id = int(user_input)
            if 0 <= cls_id < len(CLASS_NAMES):
                user_input = CLASS_NAMES[cls_id]

        # เก็บไว้ใน track ด้วย (กันหาย)
        latest_to_send["class_text"] = user_input
        latest_to_send["conf"] = 1.0

        publish_mqtt(latest_to_send["color"], user_input, 1.0)

cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
