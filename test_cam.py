import cv2

def open_camera(dev_idx=0, width=1280, height=720):
    # dev_idx=0 -> /dev/video0
    cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera /dev/video1")
        return

    # 원하는 해상도 세팅 (카메라가 지원 안 하면 무시될 수 있음)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print(f"[INFO] Opened /dev/video{dev_idx}")
    print(f"[INFO] Resolution set to {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"[INFO] FPS target {cap.get(cv2.CAP_PROP_FPS)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break

        cv2.imshow("USB Camera Preview", frame)

        # q 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit requested")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 기본적으로 /dev/video0 시도
    open_camera(dev_idx=0, width=1280, height=720)
