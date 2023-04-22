import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
lower = (0, 100, 0)
upper = (100, 255, 100)

while True:
    _, img = cap.read()
    detect = cv2.inRange(img, lower, upper)
    cv2.imshow('detect', detect)
    cv2.imshow('img', img)
    cv2.waitKey(1)