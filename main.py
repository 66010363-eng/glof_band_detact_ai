import cv2
import numpy as np

cap = cv2.VideoCapture("http://100.65.101.199:8080/video")

frame_size = 640
roi_size = 320
roi_x = (frame_size - roi_size) // 2
roi_y = (frame_size - roi_size) // 2
roi_w, roi_h = roi_size, roi_size

# ขนาดกรอบสี่เหลี่ยมคงที่
fixed_w, fixed_h = 50, 100

color_ranges = {
    # "white": ([0, 0, 200], [180, 30, 255]),
    "orange": ([5, 150, 150], [15, 255, 255]),
    "red1": ([0, 150, 150], [10, 255, 255]),
    "red2": ([170, 150, 150], [180, 255, 255]),
    "pink": ([176, 99, 244], [178, 255, 255]),
    "light_green": ([31, 221, 179], [32, 255, 255])
}

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def non_max_suppression(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)  # sort by area
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        boxes = [b for b in boxes if iou(current[:4], b[:4]) < iou_threshold]
    return keep

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_size, frame_size))
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxes = []

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                (x, y, w, h) = cv2.boundingRect(cnt)
                if roi_x < x < roi_x+roi_w and roi_y < y < roi_y+roi_h:
                    # centroid
                    cX = int(x + w/2)
                    cY = int(y + h/2)
                    # fixed box
                    top_left = (cX - fixed_w//2, cY - fixed_h//2)
                    bottom_right = (cX + fixed_w//2, cY + fixed_h//2)
                    boxes.append((top_left[0], top_left[1], fixed_w, fixed_h, area, color))

    # กรองกล่องที่ซ้อนกันด้วย NMS
    filtered_boxes = non_max_suppression(boxes, iou_threshold=0.3)

    for (x, y, w, h, area, color) in filtered_boxes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Fixed Non-Overlapping Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
