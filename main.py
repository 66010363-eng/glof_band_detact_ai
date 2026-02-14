import cv2
import numpy as np
import os
import time
import json
import paho.mqtt.client as mqtt

os.environ["OPENCV_OCL4DNN_CONFIG_PATH"] = "C:\\opencv_cache"

# Video source
cap = cv2.VideoCapture("http://100.65.101.199:8080/video")

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

# Helpers
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2]); yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = boxA[2]*boxA[3]; boxBArea = boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def non_max_suppression(boxes, iou_threshold=0.3):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        current = boxes.pop(0); keep.append(current)
        boxes = [b for b in boxes if iou(current[:4], b[:4]) < iou_threshold]
    return keep

# AI timing
last_ai_time = 0
ai_interval = 3   # run AI every 3 seconds
ai_duration = 2   # show "AI Running..." for 2 seconds
ai_running = False
ai_start_time = 0

# Latest detection result to send / label
latest_result = None  # dict: {"color","class","confidence","bbox":(x,y,w,h)}

print("Press Enter to publish latest result. If class is None and object center is inside CHECK box, you'll be prompted to input class id.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_size, frame_size))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Draw main ROI
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)

    # Draw CHECK box
    cv2.rectangle(frame, (check_x, check_y), (check_x+check_w, check_y+check_h), (0, 0, 255), 2)
    cv2.putText(frame, "CHECK", (check_x+8, check_y+36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Color detection (continuous)
    color_boxes = []
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                if roi_x < x < roi_x + roi_w and roi_y < y < roi_y + roi_h:
                    color_boxes.append((x, y, w, h, area, color_name))
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                    cv2.putText(frame, color_name, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    current_time = time.time()

    # Run AI every ai_interval seconds on detected color boxes
    if current_time - last_ai_time >= ai_interval:
        ai_running = True
        ai_start_time = current_time
        last_ai_time = current_time
        latest_result = None  # reset until new AI result found

        for (x, y, w, h, area, color_name) in color_boxes:
            roi_ball = frame[y:y+h, x:x+w]
            if roi_ball.size == 0:
                continue
            blob = cv2.dnn.blobFromImage(roi_ball, 1/255.0, (320,320), swapRB=True, crop=False)
            net.setInput(blob)
            try:
                outputs = net.forward()[0].transpose()
            except Exception:
                outputs = []

            boxes_ai = []
            for det in outputs:
                obj_score = float(det[4])
                if obj_score > 0.5:
                    class_id = int(np.argmax(det[5:]))
                    conf = float(det[5:][class_id])
                    if conf > 0.5:
                        cx, cy, bw, bh = det[0:4]
                        cx *= w; cy *= h; bw *= w; bh *= h
                        left = int(x + cx - bw/2); top = int(y + cy - bh/2)
                        boxes_ai.append((left, top, int(bw), int(bh), conf, class_id))

            filtered = non_max_suppression(boxes_ai)
            if filtered:
                # take best detection
                bx, by, bw, bh, conf, cls = filtered[0]
                cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (0,255,0), 2)
                cv2.putText(frame, f"{color_name} | cls:{cls} ({conf:.2f})", (bx, by-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                latest_result = {"color": color_name, "class": int(cls), "confidence": float(conf), "bbox": (bx,by,bw,bh)}
            else:
                # AI found nothing for this color box -> store placeholder with bbox so user can label if object in CHECK
                latest_result = {"color": color_name, "class": None, "confidence": None, "bbox": (x,y,w,h)}

    # Show AI Running status for ai_duration seconds
    if ai_running:
        cv2.putText(frame, "AI Running...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        if current_time - ai_start_time >= ai_duration:
            ai_running = False

    # If latest_result exists, draw its bbox and status near top-left
    if latest_result:
        bx, by, bw, bh = latest_result["bbox"]
        # small highlight
        cv2.rectangle(frame, (bx,by), (bx+bw, by+bh), (255,255,0), 2)
        status_text = f"Latest: color={latest_result['color']} class={latest_result['class']}"
        cv2.putText(frame, status_text, (10, frame_size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Golf Ball Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 13:  # Enter pressed -> publish latest_result (or prompt for class if missing and object center in CHECK)
        if latest_result is None:
            print("No latest result to publish.")
            continue

        # If class is None, allow user to label only if object center is inside CHECK box
        if latest_result["class"] is None:
            x0, y0, w0, h0 = latest_result["bbox"]
            cx = x0 + w0 // 2
            cy = y0 + h0 // 2
            # check center inside CHECK box
            if check_x <= cx <= check_x + check_w and check_y <= cy <= check_y + check_h:
                try:
                    user_input = input("Enter class id (integer) for the object in CHECK box: ").strip()
                    if user_input == "":
                        print("No class entered. Publish cancelled.")
                        continue
                    user_cls = int(user_input)
                    latest_result["class"] = user_cls
                    latest_result["confidence"] = 1.0  # user-labeled -> set confidence 1.0 or leave None
                except Exception as e:
                    print("Invalid input. Publish cancelled.")
                    continue
            else:
                print("Object center not inside CHECK box. Move object into CHECK box to label.")
                continue

        # Publish JSON payload
        payload = {
            "color": latest_result["color"],
            "class": int(latest_result["class"]),
            "confidence": None if latest_result["confidence"] is None else round(float(latest_result["confidence"]), 2)
        }
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))
        print("Published:", payload)

cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
mqtt_client.disconnect()
