import cv2
import numpy as np


# ======= Parameters===========
border = 7  # mm
line = 21  # mm
lower_white = (200, 200, 200)
upper_white = (255, 255, 255)
lower_black = (0, 0, 0)
upper_black = (55, 55, 55)
# =============================


pts1 = []
scale = 2
border *= scale
line *= scale
size = (border*2 + line*8)  # mm
template_size = (size, size)


def onClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts1) < 4:
            pts1.append([x, y])


def drawgrid(img):
    for i in range(border, size, line):
        cv2.line(img, (border, i), (size-border, i), (125, 125, 125), 1)
        cv2.line(img, (i, border), (i, size-border), (125, 125, 125), 1)


def drawboard(board, W, B):
    for i in range(border, size, line):
        cv2.line(board, (border, i), (size-border, i), (0, 0, 0), 1)
        cv2.line(board, (i, border), (i, size-border), (0, 0, 0), 1)
    for w in W:
        x = round((w[0] - border) / line) * line + border
        y = round((w[1] - border) / line) * line + border
        cv2.circle(board, (x, y), line//2, (255, 255, 255), -1)
    for b in B:
        x = round((b[0] - border) / line) * line + border
        y = round((b[1] - border) / line) * line + border
        cv2.circle(board, (x, y), line//2, (0, 0, 0), -1)


cv2.namedWindow('img')
cv2.setMouseCallback('img', onClick)
pts2 = [[0, 0], [template_size[1], 0], [template_size[1], template_size[0]], [0, template_size[0]]]
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
iframe = 0
kernel = np.ones((10, 10))
T = None
while True:
    _, img = cam.read()
    if len(pts1) == 4:
        T = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    if T is not None:
        img = cv2.warpPerspective(img, T, template_size)

        white = cv2.inRange(img, lower_white, upper_white)
        white = cv2.dilate(white, kernel)
        black = cv2.inRange(img, lower_black, upper_black)
        black = cv2.dilate(black, kernel)

        # White
        W = []
        contours, _ = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            if cv2.contourArea(c) > .4*line**2:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
                W.append((x+w/2, y+h/2))

        # Black
        B = []
        contours, _ = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            if cv2.contourArea(c) > .4*line**2:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 1)
                B.append((x + w / 2, y + h / 2))

        board = np.zeros_like(img)
        for i in range(3):
            board[:, :, i] = img[10, 10, i]
        drawboard(board, W, B)
        drawgrid(img)
        cv2.imshow('white', white)
        cv2.imshow('black', black)
        cv2.imshow('board', board)

    cv2.imshow('img', img)
    cv2.waitKey(1)
