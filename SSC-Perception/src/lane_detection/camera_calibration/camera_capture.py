import cv2
import os

def start_webcam():
    # img 폴더가 없으면 생성
    if not os.path.exists('img'):
        os.makedirs('img')

    cap1 = cv2.VideoCapture(0)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap1.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    frame_count = 1

    while True:
        # 웹캠으로부터 프레임 읽기
        ret1, frame1 = cap1.read()
        if not ret1:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임을 화면에 출력
        cv2.imshow('Webcam1', frame1)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF

        # 'q' 키를 누르면 루프 종료
        if key == ord('q'):
            break

        # 스페이스바를 누르면 현재 프레임을 이미지로 저장
        elif key == ord(' '):
            filename = os.path.join('img', f"{frame_count}.jpg")
            cv2.imwrite(filename, frame1)
            print(f"{filename} 저장 완료")
            frame_count += 1

    # 자원 해제
    cap1.release()
    cv2.destroyAllWindows()

# 웹캠 시작
start_webcam()
