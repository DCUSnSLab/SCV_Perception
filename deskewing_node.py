import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from rclpy.duration import Duration

class BoxStabilizerNode(Node):
    def __init__(self):
        super().__init__('box_stabilizer_node')

        # 1. 구독 (MarkerArray & Odom)
        self.sub_detections = self.create_subscription(
            MarkerArray,
            '/pcdet/detections', 
            self.marker_callback,
            10
        )
        self.sub_odom = self.create_subscription(
            Odometry,
            '/odom_bae',  # 사용하시는 odom 토픽 이름 확인 ('/odom_bae' 인가요?)
            self.odom_callback,
            10
        )

        # 2. 발행 (보정된 MarkerArray)
        self.pub_stabilized = self.create_publisher(
            MarkerArray, 
            '/pcdet/detections_stabilized', 
            10
        )

        self.last_odom = None
        self.tracked_objects = [] 
        self.next_id = 0
        
        self.alpha = 0.6  # 보정 강도 (0.0 ~ 1.0, 클수록 현재 값 신뢰, 작을수록 과거 값 신뢰)
        self.match_dist_thresh = 2.0 # 매칭 거리 (미터)

        self.get_logger().info("MarkerArray Stabilizer Initialized!")

    def odom_callback(self, msg):
        self.last_odom = msg

    def marker_callback(self, msg):
        if self.last_odom is None:
            return

        current_detections = []
        
        # MarkerArray 파싱 (중요!)
        for marker in msg.markers:
            # 삭제 명령(DELETE)이나 텍스트(TEXT_VIEW_FACING)는 무시하고 객체(CUBE, LINE 등)만 처리
            if marker.action == Marker.DELETE:
                continue
            
            # 좌표 추출
            # 주의: 일부 노드는 pose는 (0,0,0)으로 두고 points에 절대 좌표를 넣기도 함.
            # 여기서는 pose가 중심점이라고 가정합니다. (대부분의 detection node 방식)
            cx = marker.pose.position.x
            cy = marker.pose.position.y
            cz = marker.pose.position.z

            # 만약 pose가 0이고 points가 있다면 중심점 계산 (LINE_LIST 등의 경우)
            if cx == 0 and cy == 0 and len(marker.points) > 0:
                pts = np.array([(p.x, p.y, p.z) for p in marker.points])
                cx, cy, cz = np.mean(pts, axis=0)

            # 유효하지 않은 좌표(0,0,0)는 스킵 (필요 시 주석 처리)
            if cx == 0 and cy == 0 and cz == 0:
                continue

            current_detections.append({
                'x': cx,
                'y': cy,
                'z': cz,
                'raw_marker': marker # 나중에 재활용
            })

        # 1. 에고 모션 보상 (이전 박스들을 내 차 쪽으로 당겨옴)
        self.compensate_ego_motion()

        # 2. 매칭
        matched_indices = []
        for d_idx, det in enumerate(current_detections):
            best_dist = self.match_dist_thresh
            best_t_idx = -1
            
            for t_idx, track in enumerate(self.tracked_objects):
                dist = math.sqrt((det['x'] - track['x'])**2 + (det['y'] - track['y'])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_t_idx = t_idx
            
            if best_t_idx != -1:
                matched_indices.append((best_t_idx, d_idx))

        # 3. 업데이트 및 출력 메시지 생성
        output_msg = MarkerArray()
        
        # 기존 마커들 삭제 명령 추가 (Rviz 잔상 제거용)
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        output_msg.markers.append(delete_all)

        used_track_indices = set()
        used_det_indices = set()
        
        for t_idx, d_idx in matched_indices:
            used_track_indices.add(t_idx)
            used_det_indices.add(d_idx)
            
            track = self.tracked_objects[t_idx]
            det = current_detections[d_idx]
            
            # [안정화] 
            new_x = self.alpha * det['x'] + (1 - self.alpha) * track['x']
            new_y = self.alpha * det['y'] + (1 - self.alpha) * track['y']
            
            track['x'] = new_x
            track['y'] = new_y
            track['life'] = 5 
            
            # 마커 위치 수정
            out_marker = det['raw_marker']
            out_marker.id = track['id'] # ID를 우리가 관리하는 ID로 덮어쓰기 (색깔 유지 등 유리)
            
            # pose 수정
            out_marker.pose.position.x = new_x
            out_marker.pose.position.y = new_y
            
            # 만약 LINE_LIST라면 points도 전체 이동시켜야 함 (복잡하지만 단순 이동 적용)
            if len(out_marker.points) > 0:
                dx = new_x - det['x']
                dy = new_y - det['y']
                for p in out_marker.points:
                    p.x += dx
                    p.y += dy

            # 시간 업데이트 (현재 시간으로)
            out_marker.header.stamp = self.get_clock().now().to_msg()
            out_marker.lifetime = Duration(seconds=0.2).to_msg() # 잔상 방지용 짧은 수명
            
            output_msg.markers.append(out_marker)

        # 4. 신규 물체 등록
        for d_idx, det in enumerate(current_detections):
            if d_idx not in used_det_indices:
                new_obj = {
                    'id': self.next_id, # 고유 ID 부여
                    'x': det['x'],
                    'y': det['y'],
                    'z': det['z'],
                    'life': 3
                }
                # ID 충돌 방지를 위해 큰수 사용하거나 관리 필요
                # 여기서는 간단히 계속 증가시킴
                self.next_id += 1 
                if self.next_id > 10000: self.next_id = 0
                
                self.tracked_objects.append(new_obj)
                
                # 신규 마커 추가
                out_marker = det['raw_marker']
                out_marker.id = new_obj['id']
                out_marker.header.stamp = self.get_clock().now().to_msg()
                output_msg.markers.append(out_marker)

        # 5. 수명 관리
        self.tracked_objects = [t for i, t in enumerate(self.tracked_objects) 
                                if i in used_track_indices or self.decrease_life(t)]

        self.pub_stabilized.publish(output_msg)

    def decrease_life(self, track):
        track['life'] -= 1
        return track['life'] > 0

    def compensate_ego_motion(self):
        if not hasattr(self, 'prev_time'):
            self.prev_time = self.get_clock().now()
            return

        curr_time = self.get_clock().now()
        dt = (curr_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = curr_time

        # 현재 로봇 속도 (base_link 기준)
        vx = self.last_odom.twist.twist.linear.x
        vy = self.last_odom.twist.twist.linear.y
        wz = self.last_odom.twist.twist.angular.z

        # 로봇 이동만큼 물체를 반대로 이동
        cos_th = math.cos(-wz * dt)
        sin_th = math.sin(-wz * dt)

        for track in self.tracked_objects:
            # 회전
            old_x = track['x']
            old_y = track['y']
            track['x'] = old_x * cos_th - old_y * sin_th
            track['y'] = old_x * sin_th + old_y * cos_th
            
            # 이동
            track['x'] -= vx * dt
            track['y'] -= vy * dt

def main(args=None):
    rclpy.init(args=args)
    node = BoxStabilizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()