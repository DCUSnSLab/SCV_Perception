#!/usr/bin/env python3
"""
ROS2 bag → 이미지 추출 도구

사용법:
  python3 bag_to_images.py <bag_dir> [옵션]

예시:
  python3 bag_to_images.py ~/rosbag2_2024_01_01/ -i 1.0 -o ~/dataset/images
  python3 bag_to_images.py ~/rosbag2_2024_01_01/ -i 0.5 -t /camera/camera/color/image_raw

옵션:
  -i, --interval   이미지 추출 간격 (초, 기본값: 1.0)
  -o, --output     저장 디렉토리 (기본값: ./extracted_images)
  -t, --topic      카메라 토픽 (기본값: /camera/camera/color/image_raw)
  -q, --quality    JPEG 품질 0~100 (기본값: 95)
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_images(bag_dir: str, topic: str, interval: float,
                   output_dir: str, quality: int):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
    except ImportError as e:
        print(f"[오류] ROS2 환경이 필요합니다: {e}")
        print("  source /opt/ros/humble/setup.bash 후 실행하세요.")
        sys.exit(1)

    bag_path = Path(bag_dir).resolve()
    if not bag_path.exists():
        print(f"[오류] bag 디렉토리가 없습니다: {bag_path}")
        sys.exit(1)

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()

    # bag 열기
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id='sqlite3'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # 토픽 확인
    topic_types = reader.get_all_topics_and_types()
    available = [t.name for t in topic_types]
    if topic not in available:
        print(f"[오류] 토픽 '{topic}' 을 찾을 수 없습니다.")
        print(f"  bag에 있는 토픽 목록:")
        for t in available:
            print(f"    {t}")
        sys.exit(1)

    # 필터: 지정 토픽만 읽기
    filter_ = rosbag2_py.StorageFilter(topics=[topic])
    reader.set_filter(filter_)

    interval_ns = int(interval * 1e9)
    last_saved_ns = None
    saved_count = 0

    print(f"[정보] bag: {bag_path}")
    print(f"[정보] 토픽: {topic}")
    print(f"[정보] 추출 간격: {interval}초")
    print(f"[정보] 저장 경로: {out_path}")
    print(f"[정보] 추출 시작...")

    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()

        if last_saved_ns is not None and (timestamp_ns - last_saved_ns) < interval_ns:
            continue

        # 역직렬화
        msg = deserialize_message(data, Image)

        # ROS Image → OpenCV
        try:
            if msg.encoding in ('rgb8',):
                img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                img_bgr = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            print(f"[경고] 프레임 변환 실패 (ts={timestamp_ns}): {e}")
            continue

        # 파일명: frame_000001.jpg (타임스탬프 기반)
        filename = f"frame_{saved_count:06d}.jpg"
        filepath = out_path / filename
        cv2.imwrite(str(filepath), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

        saved_count += 1
        last_saved_ns = timestamp_ns

        sec = timestamp_ns / 1e9
        print(f"  [{saved_count:4d}] {filename}  (t={sec:.2f}s)", end='\r')

    print(f"\n[완료] 총 {saved_count}장 저장 → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='ROS2 bag에서 이미지를 일정 간격으로 추출합니다.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('bag_dir',
                        help='ROS2 bag 디렉토리 경로')
    parser.add_argument('-i', '--interval', type=float, default=1.0,
                        help='추출 간격 (초, 기본값: 1.0)')
    parser.add_argument('-o', '--output', default='/home/ssc/extracted_images',
                        help='저장 디렉토리 (기본값: /home/ssc/extracted_images)')
    parser.add_argument('-t', '--topic',
                        default='/camera/camera/color/image_raw',
                        help='카메라 토픽 이름')
    parser.add_argument('-q', '--quality', type=int, default=95,
                        help='JPEG 품질 0~100 (기본값: 95)')

    args = parser.parse_args()

    if not (0 < args.interval <= 60):
        print("[오류] interval은 0초 초과 60초 이하여야 합니다.")
        sys.exit(1)
    if not (0 <= args.quality <= 100):
        print("[오류] quality는 0~100 사이여야 합니다.")
        sys.exit(1)

    extract_images(
        bag_dir=args.bag_dir,
        topic=args.topic,
        interval=args.interval,
        output_dir=args.output,
        quality=args.quality,
    )


if __name__ == '__main__':
    main()
