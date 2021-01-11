# -*- coding=utf-8 -*-
import numpy as np
import cv2
import cv2.aruco as aruco

DICT_TYPE = cv2.aruco.DICT_4X4_250

def generate_marker(id, size=200, fileName='mark.png'):
    dictionary = cv2.aruco.Dictionary_get(DICT_TYPE)
    markerImage = np.zeros((size, size), dtype=np.uint8)
    # dictionary, ID (0-249), 像素值, 存储对象, 边界宽度
    markerImage = cv2.aruco.drawMarker(dictionary, id, size, markerImage, 1)
    cv2.imwrite(fileName, markerImage)

# generate_marker(0, 65, 'id0_65.png')
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(DICT_TYPE)
    arucoParameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=arucoParameters)
    if np.all(ids != None):
        height, width, _ = frame.shape
        frame = aruco.drawDetectedMarkers(frame, corners)
        x1 = (corners[0][0][0][0], corners[0][0][0][1]) # top_left
        x2 = (corners[0][0][1][0], corners[0][0][1][1]) # top_right
        x3 = (corners[0][0][2][0], corners[0][0][2][1]) # bottom_right
        x4 = (corners[0][0][3][0], corners[0][0][3][1]) # bottom_left
        print(ids[0], x1, x2, x3, x4)

    cv2.imshow('Display', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
