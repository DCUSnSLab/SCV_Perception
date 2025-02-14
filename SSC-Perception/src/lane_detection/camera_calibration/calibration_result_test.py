import numpy as np
import cv2
import glob    

with np.load('calibration_result.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    cv2.imshow('Original', frame)
    cv2.imshow('undistorted_img', undistorted_img)
    
    # 보정된 이미지 자르기
    # x, y, w, h = roi
    # dst = dst[y:y+h, x:x+w]
    # return dst
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
