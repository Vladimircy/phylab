import cv2 
import time 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


ret ,frame = cap.read()
cnt = 0
while ret:
    cnt += 1
    cv2.imwrite(f"E:\images\pic{cnt}.jpg",frame)
    del frame
    ret ,frame = cap.read()
    time.sleep(5)
