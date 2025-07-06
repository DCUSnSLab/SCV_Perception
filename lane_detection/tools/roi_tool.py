#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hexagon_picker.py
한 장의 이미지 위에서 6개의 꼭짓점을 클릭해 육각형 좌표를 추출‧시각화하는 도구
사용법: python3 hexagon_picker.py --image /path/to/image.jpg
"""
import argparse
import cv2
import sys

# ────────── 파라미터 파싱 ──────────
parser = argparse.ArgumentParser(description="Click 6 points to define a hexagon")
parser.add_argument("--image", "-i", required=True, help="Path to input image")
args = parser.parse_args()

# ────────── 이미지 읽기 ──────────
img = cv2.imread(args.image)
if img is None:
    sys.exit(f"이미지를 불러올 수 없습니다: {args.image}")

orig = img.copy()        # 원본 보존
points = []               # 클릭된 좌표 저장


def draw_overlay(frame):
    """현재 points 상태를 화면에 그려 준다."""
    # 점
    for p in points:
        cv2.circle(frame, p, 6, (0, 255, 0), -1, cv2.LINE_AA)

    # 육각형 윤곽
    if len(points) == 6:
        for i in range(6):
            cv2.line(frame, points[i], points[(i + 1) % 6], (255, 0, 0), 2, cv2.LINE_AA)


def on_mouse(event, x, y, flags, _):
    global points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 6:
            points.append((x, y))
            print(f"[{len(points)}/6] 클릭: ({x}, {y})")
        else:
            print("이미 6개를 모두 선택했습니다. 'r' 키로 초기화 후 다시 시도하세요.")

        img = orig.copy()
        draw_overlay(img)


cv2.namedWindow("Hexagon Picker")
cv2.setMouseCallback("Hexagon Picker", on_mouse)

print("\n▶ 창이 뜨면 육각형 꼭짓점을 왼쪽(위) → 왼쪽(중간) → 왼쪽(아래) → 오른쪽(위) → 오른쪽(중간) → 오른쪽(아래) 순으로 클릭하세요.")
print("▶ 'r' 키: 다시 시작 | 'q' 또는 ESC: 종료\n")

while True:
    cv2.imshow("Hexagon Picker", img)
    key = cv2.waitKey(20) & 0xFF

    # 초기화
    if key == ord("r"):
        points.clear()
        img = orig.copy()
        print("\n좌표 리스트가 초기화되었습니다. 다시 클릭하세요.")

    # 종료
    elif key in (ord("q"), 27):  # 27=ESC
        break

# ────────── 출력 ──────────
cv2.destroyAllWindows()
if len(points) == 6:
    print("\n=== 최종 육각형 좌표 ===")
    print("hexagon_points = [")
    for p in points:
        print(f"    {p},")
    print("]")
else:
    print("\n6개 모두 찍지 않아 좌표가 완성되지 않았습니다.")
