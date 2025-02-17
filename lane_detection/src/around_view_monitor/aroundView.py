import numpy as np
import cv2

def draw_vehicle(canvas, car_width_px, car_length_px): # 임시 자동차 그리기
    car_color = (255, 0, 0)
    canvas_height, canvas_width = canvas.shape[:2]
    top_left = (canvas_width // 2 - car_width_px // 2, canvas_height // 2 - car_length_px // 2)
    bottom_right = (top_left[0] + car_width_px, top_left[1] + car_length_px)
    cv2.rectangle(canvas, top_left, bottom_right, car_color, -1)
    return top_left, bottom_right

def setup_camera(width, height, camNum, calibration_data): # 카메라 해상도, cailbration 데이터
    cap = cv2.VideoCapture(camNum, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise Exception("Error: Could not open webcam.")
    with np.load(calibration_data) as data:
        mtx = data['mtx']
        dist = data['dist']
    return cap, mtx, dist

def transform_to_bev(image, src_points, dst_points, output_size): # bev 변환
    width = int(max(dst_points[:, 0]) - min(dst_points[:, 0]))
    height = int(max(dst_points[:, 1]) - min(dst_points[:, 1]))
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, M, (width, height))
    return bev_image

def imgChange(frame, src_points, dst_points, mtx, dist, canvas_width = 1000, canvas_height = 1000): # cali img, bev img
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistort_img = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    undistort_img = undistort_img[y:y+h, x:x+w]
    bev_image = transform_to_bev(undistort_img, src_points, dst_points, (canvas_width, canvas_height))
    return undistort_img, bev_image

def draw_points(image, points): # bev 위한 4점 확인
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
    return image

def main():
    car_width_px = 280
    car_length_px = 280
    aroundView_height = 600
    aroundView_width = 600
    aroundView = np.zeros((aroundView_height, aroundView_width, 3), dtype=np.uint8)
    
    car_top_left, car_bottom_right = draw_vehicle(aroundView, car_width_px, car_length_px)

    front_cam , front_mtx, front_dist= setup_camera(1280, 720, 1, '../calibration_result/back_cam_cali_result.npz')
    front_cam_src_points = np.float32([[-300, 666], [1517, 666], [1217, 340], [0, 340]])

    back_cam , back_mtx, back_dist= setup_camera(1280, 720, 0, '../calibration_result/back_cam_cali_result.npz')
    back_cam_src_points = np.float32([[-300, 666], [1517, 666], [1217, 340], [0, 340]])

    # dst_points = np.float32([[0, aroundView_height], [aroundView_width, aroundView_height], [aroundView_width, 0], [0, 0]])
    dst_points = np.float32([[0, 460], [1280, 460], [1280, 0], [0, 0]])
    while True:
        front_ret, front_frame = front_cam.read()
        back_ret, back_frame = back_cam.read()

        if not front_ret or not back_ret:
            break

        front_undistort_img, front_bev_img = imgChange(front_frame, front_cam_src_points, dst_points, front_mtx, front_dist)
        back_undistort_img, back_bev_img = imgChange(back_frame, back_cam_src_points, dst_points, back_mtx, back_dist)

        front_bev_img_cut = front_bev_img[-400:, :]
        back_bev_img_cut = cv2.flip(back_bev_img[-400:,:], 0)

        front_bev_img_cut_resized = cv2.resize(front_bev_img_cut, (front_bev_img_cut.shape[1] // 3, front_bev_img_cut.shape[0] // 3))
        back_bev_img_cut_resized = cv2.resize(back_bev_img_cut, (back_bev_img_cut.shape[1] // 3, back_bev_img_cut.shape[0] // 3))

        top_padding = car_top_left[1] - front_bev_img_cut_resized.shape[0]
        top_padding = max(top_padding, 0)
        bottom_padding = car_bottom_right[1] + back_bev_img_cut_resized.shape[0]

        front_padding_left = (aroundView_width - front_bev_img_cut_resized.shape[1]) // 2
        back_padding_left = (aroundView_width - back_bev_img_cut_resized.shape[1]) // 2

        aroundView[top_padding:top_padding + front_bev_img_cut_resized.shape[0], front_padding_left:front_padding_left + front_bev_img_cut_resized.shape[1]] = front_bev_img_cut_resized
        aroundView[car_bottom_right[1]:bottom_padding, back_padding_left:back_padding_left + back_bev_img_cut_resized.shape[1]] = back_bev_img_cut_resized

        cv2.imshow('AVM', aroundView)
        cv2.imshow('front_bev', front_bev_img)
        cv2.imshow('back_bev', back_bev_img)
        cv2.imshow('back_img', back_undistort_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            front_cam.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
