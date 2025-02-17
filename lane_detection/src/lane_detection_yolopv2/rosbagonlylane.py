import argparse
import time
from pathlib import Path
import cv2
import torch
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from utils.utils import time_synchronized, select_device, increment_path, scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, AverageMeter

class LoadImagesFromBag:
    def __init__(self, bag_file, topic_name, img_size=640, stride=32):
        self.bag_file = bag_file
        self.topic_name = topic_name
        self.img_size = img_size
        self.stride = stride
        self.bridge = CvBridge()
        self.bag = rosbag.Bag(bag_file, 'r')
        self.messages = self.bag.read_messages(topics=[topic_name])
        self.iter_messages = iter(self.messages)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            topic, msg, t = next(self.iter_messages)
            if topic == self.topic_name:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                img_resized = cv2.resize(img, (640, 384))
                return None, img_resized, img, None
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV image: {e}")
        except StopIteration:
            self.bag.close()
            raise StopIteration

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--bag-file', type=str, default='', help='path to the rosbag file')
    parser.add_argument('--topic-name', type=str, default='', help='name of the topic to read images from')
    return parser

def detect():
    # 설정 및 디렉토리 생성
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # 이미지를 저장할지 여부 결정

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 실행 횟수 증가
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 디렉토리 생성

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision은 CUDA에서만 지원
    model = model.to(device)

    if half:
        model.half()  # FP16으로 변환
    model.eval()

    # 데이터 로더 설정
    vid_path, vid_writer = None, None
    if opt.bag_file and opt.topic_name:
        dataset = LoadImagesFromBag(opt.bag_file, opt.topic_name, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 한 번 실행
    t0 = time.time()
        # fps 계산
    frame_count = 0
    start_time = time.time()
    for path, img, im0s, vid_cap in dataset:
        cv2.imshow('Resized Image', img)
        cv2.waitKey(1)
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1)  # [channels, height, width]로 변환
        #print(img)
        img = img.half() if half else img.float()  # uint8에서 fp16/32로 변환
        img /= 255.0  # 0 - 255를 0.0 - 1.0으로 변환

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # 추가 시간 소모
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # NMS 적용
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        black_background = np.zeros_like(im0s)
        black_lane = np.zeros_like(im0s)
        da_seg_mask_colored = np.zeros_like(im0s)
        ll_seg_mask_colored = np.zeros_like(im0s)
        da_seg_mask_colored[da_seg_mask == 1] = [0, 255, 0]  # for driving area
        ll_seg_mask_colored[ll_seg_mask == 1] = [255, 255, 255]
        combined_mask = cv2.addWeighted(da_seg_mask_colored, 0.5, ll_seg_mask_colored, 0.5, 0)
        black_background = cv2.addWeighted(black_background, 1, combined_mask, 1, 0)
        black_lane = cv2.addWeighted(black_lane, 1, ll_seg_mask_colored, 1, 0)
        img_height, img_width = im0s.shape[:2]
        # 프레임 카운트 증가
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time>1:
            fps = frame_count
            frame_count = 0
            elapsed_time = 0
            start_time = time.time()
            print(f"FPS: {fps:.2f}")
        for i, det in enumerate(pred):  # 이미지당 디텍션
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if p is not None:
                p = Path(p)  # Path로 변환
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # 출력 문자열
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # whwh로 정규화 게인
                if len(det):
                    # 박스를 원본 이미지 크기로 조정
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # 결과 출력
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # 클래스별 디텍션 수

                    # 결과 저장
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # 파일에 저장
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 정규화된 xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # 라벨 포맷
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img:  # 이미지에 박스 추가
                            plot_one_box(xyxy, im0, line_thickness=3)

            # 시간 출력 (추론)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            cv2.imshow('1', black_background)
            cv2.waitKey(1)
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            cv2.imshow('2', im0)
            cv2.imshow('3', black_lane)
            # 결과 저장 (디텍션이 포함된 이미지)
            if save_img and p is not None:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' 또는 'stream'
                    if vid_path != save_path:  # 새로운 비디오
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # 이전 비디오 라이터 해제
                        if vid_cap:  # 비디오
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 스트림
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
