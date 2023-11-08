from ultralytics import YOLO
import cv2 
import time 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = YOLO("best.pt") 

ret ,frame = cap.read()
pred = 0
while ret:
    res = model(frame)[0]
    boxes = res.boxes.xyxy
    box_index = 0
    for cood in boxes:
        pic = frame[int(cood[1]):int(cood[3]),int(cood[0]):int(cood[2])]
        pic = cv2.resize(pic,((int(cood[2])-int(cood[0]))*4,(int(cood[3])-int(cood[1]))*4))
        cv2.imwrite(f"E:\\datasets\\result\\result_pred{pred}_boxindex{box_index}.jpg",pic)
        box_index += 1
    pred += 1
    time.sleep(2)
    ret ,frame = cap.read()

