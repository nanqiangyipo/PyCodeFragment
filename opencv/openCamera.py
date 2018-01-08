import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while(True):
        statu, frame = cap.read()
        cv2.imshow('czm', frame)
        key = cv2.waitKey(1)
        if key==113:
            cap.release()
            break

