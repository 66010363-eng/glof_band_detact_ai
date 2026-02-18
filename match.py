import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret: break

    # ตั้งค่า ROI กลางภาพ
    height, width = frame.shape[:2]
    roi_w, roi_h = 200, 200  # ปรับขนาดให้เหมาะกับบรรทัดตัวอักษร
    x1, y1 = int((width - roi_w) / 2), int((height - roi_h) / 2)
    x2, y2 = x1 + roi_w, y1 + roi_h

    # 1. เตรียมภาพ ROI
    roi_img = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # 2. ใช้ Threshold แบบ Adaptive (ช่วยให้จับตัวอักษรได้ดีขึ้นแม้แสงไม่เท่ากัน)
    # วัตถุสีดำจะกลายเป็นสีขาว (255) พื้นหลังเป็นดำ (0)
    thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 10)

    # 3. Morphology: เชื่อมตัวอักษรที่อยู่ใกล้กันให้เป็นกลุ่ม (Dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # 4. หา Contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ภาพที่จะแสดงผล (ทำเป็นขาวดำ)
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        # --- ส่วนการกรอง (Filtering) ---
        # กรองทิ้งถ้า:
        # 1. เล็กเกินไป (Area < 500)
        # 2. ใหญ่เกินไป (Area > 20000) หรือกว้าง/สูง เกิน 80% ของ ROI
        min_area, max_area = 500, 20000
        max_w, max_h = roi_w * 0.8, roi_h * 0.8

        if min_area < area < max_area and bw < max_w and bh < max_h:
            # วาดสี่เหลี่ยมล้อมรอบเฉพาะตัวที่ผ่านการกรอง
            cv2.rectangle(display_frame, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (255), 2)

            cv2.putText(display_frame, "Target", (x1 + bx, y1 + by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)

    # วาดกรอบ ROI สีดำ
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0), 2)

    cv2.imshow('Text/Logo Detection ROI', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()