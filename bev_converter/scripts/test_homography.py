#!/usr/bin/env python3
"""
BEV 변환 코드 테스트를 위한 스크립트
호모그래피 변환과 좌표계 변환 테스트
"""
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys

# 현재 스크립트의 디렉토리 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
# config 디렉토리 경로
config_path = os.path.join(os.path.dirname(script_dir), 'config', 'camera_params.yaml')

def load_camera_params(config_file):
    """카메라 파라미터 로드"""
    with open(config_file, 'r') as f:
        params = yaml.safe_load(f)
    return params.get('camera', {})

def compute_homography_inv(intrinsics, extrinsics):
    """
    호모그래피 계산 함수
    """
    # 내부 파라미터 추출
    fx = intrinsics.get("fx", 1.0)
    fy = intrinsics.get("fy", 1.0)
    cx = intrinsics.get("cx", 0.0)
    cy = intrinsics.get("cy", 0.0)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    # 외부 파라미터 추출
    R_list = extrinsics.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    R = np.array(R_list).reshape(3, 3)
    h = extrinsics.get("translation", [0, 0, 0.45])[2]
    t_vec = np.array(extrinsics.get("translation", [0, 0, h]))
    
    # 호모그래피 계산
    r1 = R[:, 0].reshape(3, 1)
    r2 = R[:, 1].reshape(3, 1)
    t = t_vec.reshape(3, 1)
    
    H = np.dot(K, np.hstack((r1, r2, t)))
    
    try:
        H_inv = np.linalg.inv(H)
        return H_inv, h
    except np.linalg.LinAlgError:
        print("호모그래피 행렬이 특이해서 역행렬을 계산할 수 없습니다.")
        return None, h

def image_to_ground(u, v, H_inv):
    """
    이미지 좌표를 지면 좌표로 변환
    """
    # 동차 좌표로 변환
    point_img = np.array([u, v, 1.0]).reshape(3, 1)
    
    # 역호모그래피 적용
    ground_point_h = np.dot(H_inv, point_img)
    
    # 정규화
    if abs(ground_point_h[2, 0]) < 1e-10:
        print("정규화 오류: 0으로 나누기 또는 매우 작은 값")
        return 0.0, 0.0
    
    ground_point_h /= ground_point_h[2, 0]
    X = ground_point_h[0, 0]
    Y = ground_point_h[1, 0]
    
    return X, Y

def ground_to_rviz(X, Y, scale_factor=1.0):
    """
    지면 좌표를 RViz 좌표로 변환
    """
    X_scaled = X * scale_factor
    Y_scaled = Y * scale_factor
    
    # 기존 코드 방식 (문제가 있는 방식)
    X_old = Y * 10
    Y_old = -X * 10
    
    return X_scaled, Y_scaled, X_old, Y_old

def ground_to_image(X, Y, Z, camera_matrix, R, t):
    """
    지면 좌표를 이미지 좌표로 변환
    ※ 호모그래피의 역방향 변환
    """
    # 지면 좌표 (3D)
    world_point = np.array([[X], [Y], [Z], [1.0]])
    
    # 회전 및 이동 행렬
    Rt = np.hstack((R, t.reshape(3, 1)))
    
    # 월드 좌표를 카메라 좌표로 변환
    camera_point = np.dot(Rt, world_point)
    
    # 카메라 좌표를 이미지 좌표로 변환
    if camera_point[2, 0] < 1e-10:  # Z가 0에 가까우면
        return None
    
    # 이미지 평면에 투영
    point_2d = np.dot(camera_matrix, camera_point[:3])
    point_2d /= point_2d[2, 0]
    
    u = float(point_2d[0, 0])
    v = float(point_2d[1, 0])
    
    return u, v

def test_homography_with_grid():
    """
    그리드 패턴을 사용한 호모그래피 테스트
    """
    camera_params = load_camera_params(config_path)
    intrinsics = camera_params.get('intrinsics', {})
    extrinsics = camera_params.get('extrinsics', {})
    
    H_inv, h = compute_homography_inv(intrinsics, extrinsics)
    if H_inv is None:
        return
    
    # 이미지 중심
    img_width = 1920  # 가정
    img_height = 1080  # 가정
    
    # 이미지 좌표 그리드 생성 (아래쪽 절반만 사용)
    grid_step = 100
    u_coords = np.arange(0, img_width, grid_step)
    v_coords = np.arange(img_height//2, img_height, grid_step)
    
    # 지면 좌표와 RViz 좌표를 저장할 리스트
    ground_points = []
    rviz_points_new = []
    rviz_points_old = []
    
    scale_factor = 1.0 / h  # 카메라 높이 기반 스케일 팩터
    
    for u in u_coords:
        for v in v_coords:
            # 이미지 좌표를 지면 좌표로 변환
            X, Y = image_to_ground(u, v, H_inv)
            
            # 거리에 따른 보정 (옵션)
            distance = np.sqrt(X*X + Y*Y)
            if distance > 5.0:
                correction = 5.0 / distance
                X *= correction
                Y *= correction
            
            # 지면 좌표를 RViz 좌표로 변환
            X_rviz_new, Y_rviz_new, X_rviz_old, Y_rviz_old = ground_to_rviz(X, Y, scale_factor)
            
            ground_points.append((X, Y))
            rviz_points_new.append((X_rviz_new, Y_rviz_new))
            rviz_points_old.append((X_rviz_old, Y_rviz_old))
    
    # 한글 폰트 문제 해결을 위해 영문만 사용
    # 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 지면 좌표 (원본)
    X_points, Y_points = zip(*ground_points)
    ax1.scatter(X_points, Y_points, c='blue', marker='.', label='Ground Coordinates')
    ax1.set_title('Original Ground Coordinates')
    ax1.set_xlabel('X (Camera Forward)')
    ax1.set_ylabel('Y (Camera Left)')
    ax1.grid(True)
    ax1.axis('equal')
    
    # RViz 좌표 비교
    X_new, Y_new = zip(*rviz_points_new)
    X_old, Y_old = zip(*rviz_points_old)
    
    ax2.scatter(X_new, Y_new, c='green', marker='.', label='Improved Method')
    ax2.scatter(X_old, Y_old, c='red', marker='x', label='Original Method')
    ax2.set_title('RViz Coordinate Comparison')
    ax2.set_xlabel('X (RViz)')
    ax2.set_ylabel('Y (RViz)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    
    # 결과 저장
    output_dir = os.path.dirname(script_dir)
    plt.savefig(os.path.join(output_dir, 'homography_test.png'))
    print(f"테스트 결과가 {os.path.join(output_dir, 'homography_test.png')}에 저장되었습니다.")
    
    # 스케일 비교
    print(f"\n카메라 높이: {h} 미터")
    print(f"제안된 스케일 팩터: {scale_factor}")
    print("\n좌표 예시 (이미지 중심 하단):")
    
    # 이미지 중심 하단 좌표
    center_u = img_width // 2
    center_v = img_height
    
    # 변환 예시
    X, Y = image_to_ground(center_u, center_v, H_inv)
    X_rviz_new, Y_rviz_new, X_rviz_old, Y_rviz_old = ground_to_rviz(X, Y, scale_factor)
    
    print(f"이미지 좌표: ({center_u}, {center_v})")
    print(f"지면 좌표: ({X:.3f}, {Y:.3f})")
    print(f"새 RViz 좌표: ({X_rviz_new:.3f}, {Y_rviz_new:.3f})")
    print(f"기존 RViz 좌표: ({X_rviz_old:.3f}, {Y_rviz_old:.3f})")
    
    return scale_factor

def test_distance_accuracy():
    """
    거리 정확도 테스트 - 개선된 버전
    지면 위 특정 거리에 있는 객체의 좌표 변환 정확도 확인
    """
    camera_params = load_camera_params(config_path)
    intrinsics = camera_params.get('intrinsics', {})
    extrinsics = camera_params.get('extrinsics', {})
    
    H_inv, h = compute_homography_inv(intrinsics, extrinsics)
    if H_inv is None:
        return
    
    # 이미지 크기 및 중심
    img_width = 1920
    img_height = 1080
    
    # 이미지 중심선에서 다양한 거리에 있는 포인트 테스트
    test_distances = [1, 2, 3, 5, 10, 15, 20]  # 미터 단위
    
    # 카메라 파라미터 추출
    fx = intrinsics.get("fx", 1.0)
    fy = intrinsics.get("fy", 1.0)
    cx = intrinsics.get("cx", 0.0)
    cy = intrinsics.get("cy", 0.0)
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # 회전 및 이동 행렬
    R_list = extrinsics.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    R = np.array(R_list).reshape(3, 3)
    t_vec = np.array(extrinsics.get("translation", [0, 0, h]))
    
    # 결과 저장
    results = []
    
    # 월드 좌표에서 이미지 좌표로의 변환 테스트
    fig, ax = plt.subplots(figsize=(10, 6))
    world_points = []
    image_points = []
    recovered_points = []
    
    for dist in test_distances:
        # 지면 좌표 (카메라 앞쪽 dist 미터 지점, Z=0 평면)
        X_ground = dist
        Y_ground = 0.0
        Z_ground = 0.0  # 지면 평면
        
        # 지면 좌표를 이미지 좌표로 변환
        img_result = ground_to_image(X_ground, Y_ground, Z_ground, camera_matrix, R, t_vec)
        if img_result is None:
            print(f"Warning: Could not project point at distance {dist}m to image")
            continue
            
        u, v = img_result
        
        # 이미지 좌표를 다시 지면 좌표로 변환 (호모그래피 이용)
        # 이 과정은 지면 좌표 -> 이미지 좌표 -> 지면 좌표 반복
        X_back, Y_back = image_to_ground(u, v, H_inv)
        
        # 거리 오차 계산
        distance_error = abs(X_back - X_ground)
        error_percent = (distance_error / dist) * 100.0 if dist > 0 else 0.0
        
        # 그래프용 데이터 저장
        world_points.append((X_ground, Y_ground))
        image_points.append((u, v))
        recovered_points.append((X_back, Y_back))
        
        # 결과 저장
        results.append({
            'expected_dist': X_ground,
            'image_coord': (u, v),
            'computed_dist': X_back,
            'error': distance_error,
            'error_percent': error_percent
        })
    
    # 결과 그래프 생성
    if world_points and recovered_points:
        # 월드 좌표와 복원된 좌표 비교
        world_x, world_y = zip(*world_points)
        rec_x, rec_y = zip(*recovered_points)
        
        ax.scatter(world_x, world_y, color='blue', marker='o', label='Original World Points')
        ax.scatter(rec_x, rec_y, color='red', marker='x', label='Recovered World Points')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Coordinate Transformation Accuracy')
        ax.grid(True)
        ax.legend()
        
        # 그래프 저장
        output_dir = os.path.dirname(script_dir)
        plt.savefig(os.path.join(output_dir, 'distance_accuracy_test.png'))
    
    # 결과 출력
    print("\nDistance Accuracy Test Results:")
    print("Expected Distance(m) | Image Coords (u,v) | Computed Distance(m) | Error(m) | Error(%)")
    print("-" * 90)
    
    for r in results:
        u, v = r['image_coord']
        print(f"{r['expected_dist']:20.1f} | ({u:6.1f}, {v:6.1f}) | {r['computed_dist']:18.3f} | {r['error']:7.3f} | {r['error_percent']:7.2f}")
    
    # 스케일 팩터 계산 (카메라 높이 기준)
    scale_factor = 1.0 / h
    return scale_factor

if __name__ == "__main__":
    print("BEV 변환 테스트 시작...")
    try:
        # 그리드 테스트
        scale_factor_grid = test_homography_with_grid()
        print("\n그리드 테스트 완료!")
        
        # 거리 정확도 테스트
        scale_factor_dist = test_distance_accuracy()
        print("\n거리 정확도 테스트 완료!")
        
        print(f"\n권장 스케일 팩터: {scale_factor_grid}")
        print(f"이 값을 bev_node.py의 yolo_callback 함수에서 scale_factor 변수에 할당하세요.")
        print("예: scale_factor = {:.3f}  # 카메라 높이의 역수로 스케일링".format(scale_factor_grid))
        
        print("\n테스트 완료!")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
