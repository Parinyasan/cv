import cv2
import numpy as np


def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts1) < 4:
            pts1.append([x, y])



area = (300, 300)
pts1 = []
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('img')
cv2.setMouseCallback('img', onClick)
pts2 = [(0, 0), (area[1], 0), (area[1], area[0]), (0, area[0])]
cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)

T = None
while True:
    _, img = cam.read()
    if len(pts1) == 4:
        T = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    if T is not None:
        img = cv2.warpPerspective(img, T, area)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bw = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        kernel = np.ones((11, 11))
        bw = cv2.erode(bw, kernel)
        bw = cv2.dilate(bw, kernel)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        for cnt in contours:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if 10 < radius < 100:
                center = (int(x), int(y))
                radius = int(radius)
                img = cv2.circle(img, center, radius, (0, 255, 0), 2)
                img = cv2.putText(img, str(radius), center, font, .3, (0, 255, 0), 1, cv2.LINE_AA)
                if 18 <= radius <= 24:
                    img = cv2.putText(img, '1', (int(x), int(y) + 10), font, .3, (255, 0, 0), 1, cv2.LINE_AA)
                    count += 1
                if 26 <= radius <= 30:
                    img = cv2.putText(img, '10', (int(x), int(y) + 10), font, .3, (255, 0, 0), 1, cv2.LINE_AA)
                    count += 10
        img = cv2.putText(img, str(count) + ' Baht', (10, 10), font, .3, (255, 0, 0), 1, cv2.LINE_AA)

    if T is not None:
        cv2.imshow('bw', bw)
    else:
        for pt in pts1:
            cv2.circle(img, (pt[0], pt[1]), 5, (0, 255, 0), -1)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
