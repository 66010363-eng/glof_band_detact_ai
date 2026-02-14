import cv2
import numpy as np

# เปิดกล้องหรือ IP Camera
cap = cv2.VideoCapture("http://100.65.101.199:8080/video")

def show_hsv_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # คลิกซ้าย
        frame = param["frame"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        pixel = hsv[y, x]  # ค่า HSV ของพิกเซลที่คลิก
        print(f"ตำแหน่ง ({x},{y}) HSV = {pixel}")

        # แสดงค่า HSV บนภาพ
        cv2.putText(frame, f"HSV:{pixel}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.namedWindow("Check HSV")
param = {"frame": None}
cv2.setMouseCallback("Check HSV", show_hsv_value, param)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))  # ปรับขนาดมาตรฐาน YOLOv8n
    param["frame"] = frame

    cv2.imshow("Check HSV", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()
