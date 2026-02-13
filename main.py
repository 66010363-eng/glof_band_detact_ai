import cv2
import numpy as np
from ultralytics import YOLO
import time

# =========================
# CONFIG
# =========================
STREAM_URL = "http://10.20.13.23:8080/video"
MODEL_PATH = "my_model.onnx"

FRAME_W = 640
FRAME_H = 480
MIN_AREA = 800
IOU_THRESHOLD = 0.4
YOLO_EVERY_N_FRAME = 2   # ลด inference frequency

# Detection Zone (กรอบใหญ่)
ZONE_X, ZONE_Y = 100, 80
ZONE_W, ZONE_H = 440, 320

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# GLOBAL
# =========================
bg_color =None
select_bg_mode = True
track_boxes = []
frame_count = 0

# =========================
# IOU
# =========================
def compute_iou(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    xi1, yi1 = max(x1,x2), max(y1,y2)
    xi2, yi2 = min(x1+w1,x2+w2), min(y1+h1,y2+h2)

    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union>0 else 0

# =========================
# COLOR TRACK REFINE
# =========================
def refine_box(frame, box):
    x,y,w,h = box
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return box

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    lower = np.array([
        max(0, bg_color[0] - 15),
        max(0, bg_color[1] - 20),
        max(0, bg_color[2] - 20)
    ], dtype=np.uint8)

    upper = np.array([
        min(255, bg_color[0] + 15),
        min(255, bg_color[1] + 20),
        min(255, bg_color[2] + 20)
    ], dtype=np.uint8)

    lower = np.clip(lower, 0, 255)
    upper = np.clip(upper, 0, 255)

    mask = cv2.inRange(lab, lower, upper)
    mask = cv2.bitwise_not(mask)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 200:
            rx,ry,rw,rh = cv2.boundingRect(cnt)
            return (x+rx, y+ry, rw, rh)

    return box

# =========================
# MOUSE
# =========================
def mouse_callback(event,x,y,flags,param):
    global bg_color, select_bg_mode, track_boxes

    frame = param

    if event == cv2.EVENT_LBUTTONDOWN and select_bg_mode:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        bg_color = lab[y,x]
        select_bg_mode = False
        return

    if event == cv2.EVENT_LBUTTONDOWN and not select_bg_mode:
        track_boxes.append((x-40,y-40,80,80))

    if event == cv2.EVENT_RBUTTONDOWN:
        for box in track_boxes[:]:
            bx,by,bw,bh = box
            if bx<x<bx+bw and by<y<by+bh:
                track_boxes.remove(box)
                break

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(STREAM_URL)
cv2.namedWindow("Tracking")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(FRAME_W,FRAME_H))
    cv2.setMouseCallback("Tracking", mouse_callback, frame)

    if select_bg_mode:
        cv2.putText(frame,"Click to select BG",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.imshow("Tracking",frame)
        if cv2.waitKey(1)==ord('q'):
            break
        continue

    # =========================
    # DRAW DETECTION ZONE
    # =========================
    cv2.rectangle(frame,(ZONE_X,ZONE_Y),
                  (ZONE_X+ZONE_W,ZONE_Y+ZONE_H),
                  (0,255,255),2)

    zone = frame[ZONE_Y:ZONE_Y+ZONE_H,
                 ZONE_X:ZONE_X+ZONE_W]

    # =========================
    # AUTO DETECT INSIDE ZONE
    # =========================
    lab = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)

    lower = np.clip([bg_color[0]-15,
                     bg_color[1]-20,
                     bg_color[2]-20],0,255)

    upper = np.clip([bg_color[0]+15,
                     bg_color[1]+20,
                     bg_color[2]+20],0,255)

    mask = cv2.inRange(lab,np.array(lower),np.array(upper))
    mask = cv2.bitwise_not(mask)

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    auto_boxes = []

    for cnt in contours:
        if cv2.contourArea(cnt)>MIN_AREA:
            x,y,w,h = cv2.boundingRect(cnt)
            new_box = (x+ZONE_X,y+ZONE_Y,w,h)

            duplicate=False
            for old in auto_boxes:
                if compute_iou(new_box,old)>IOU_THRESHOLD:
                    duplicate=True
                    break

            if not duplicate:
                auto_boxes.append(new_box)

    all_boxes = auto_boxes + track_boxes

    updated_boxes=[]
    frame_count+=1

    for box in all_boxes:
        box = refine_box(frame,box)
        updated_boxes.append(box)

        x, y, w, h = box

        roi = frame[y:y + h, x:x + w]

        if roi.size == 0:
            continue

        roi_resized = cv2.resize(roi, (640, 640))

        results = model(
            roi_resized,
            device=0,  # บังคับ GPU
            half=True,
            verbose=False
        )

        for r in results:
            for b in r.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"

                cv2.putText(frame, label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    track_boxes=updated_boxes

    # =========================
    # FPS
    # =========================
    now=time.time()
    fps=1/(now-prev_time)
    prev_time=now

    cv2.putText(frame,f"FPS: {int(fps)}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,(0,255,0),2)

    cv2.imshow("Tracking",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    if key==ord('b'):
        select_bg_mode=True
        track_boxes.clear()

cap.release()
cv2.destroyAllWindows()
