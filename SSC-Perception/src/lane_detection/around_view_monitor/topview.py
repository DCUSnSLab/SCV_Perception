import numpy as np
import cv2

def transform_to_bev(image, src_points, dst_points):
    width = int(max(dst_points[:, 0]) - min(dst_points[:, 0]))
    height = int(max(dst_points[:, 1]) - min(dst_points[:, 1]))
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, M, (width, height))
    return bev_image

def draw_points(image, points):
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)  # 파란색 점
    return image

# 마우스 클릭 이벤트 처리 함수
def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        param['points'].append((x, y))
        param['image'] = draw_points(param['image'], [(x, y)])
        cv2.imshow('undistorted_img', param['image'])

with np.load('calibration_result/back_cam_cali_result.npz') as data:
    mtx = data['mtx']
    dist = data['dist']

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

src_points = np.float32([[-300, 666], [1517, 666], [1217, 340], [0, 340]])  # 왼쪽아래, 오른쪽아래, 오른쪽위, 왼쪽위
dst_points = np.float32([[0, 460], [1280, 460], [1280, 0], [0, 0]])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted_img = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    print(undistorted_img.shape[:2])
    img_undistort_point = draw_points(undistorted_img.copy(), src_points)
    bev_image = transform_to_bev(undistorted_img, src_points, dst_points)

    # 이미지를 먼저 창에 표시
    cv2.imshow('Original', frame)
    cv2.imshow('undistorted_img', img_undistort_point)
    cv2.imshow('bev_image', bev_image)

    # 마우스 클릭 이벤트 콜백 설정
    mouse_params = {'points': [], 'image': img_undistort_point}
    cv2.setMouseCallback('undistorted_img', on_mouse_click, mouse_params)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
