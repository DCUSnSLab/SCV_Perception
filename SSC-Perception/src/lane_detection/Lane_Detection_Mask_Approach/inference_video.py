import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import os
import time
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

class InferenceNode:
    def __init__(self):
        rospy.init_node('lane_other_detection_node', anonymous=True)
        self.bridge = CvBridge()
        self.image = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.rate = rospy.Rate(10)  # 10 Hz
        self.subscriber = rospy.Subscriber('/zed_node/left/image_rect_color', Image, self.callback)
        self.show = args.show
        if self.show:
            cv2.namedWindow('Result', cv2.WINDOW_NORMAL)

    def load_model(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False, num_classes=len(class_names))
        model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
        model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
        model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))

        # Load model weights
        ckpt = torch.load(args.weights)
        model.load_state_dict(ckpt['model'])
        
        # self.device를 사용하여 모델을 GPU 또는 CPU에 로드
        model.to(self.device).eval()
        return model

    def callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.run_inference(self.image)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting ROS Image message to OpenCV image: {e}")

    def run_inference(self, frame):
        # Convert OpenCV BGR image to PIL RGB image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = PILImage.fromarray(image)
        orig_image = image.copy()

        # Transform the image and move to the device
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Get model outputs
        masks, boxes, labels = get_outputs(image, self.model, args.threshold)

        # Filter the classes as needed
        masks, boxes, labels = self.filter_class(masks, boxes, labels)

        # Create a black background image
        black_background = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # Draw the segmentation map on the original image (if desired)
        result = draw_segmentation_map(orig_image, masks, boxes, labels, args)
        black_result = draw_segmentation_map(black_background, masks, boxes, labels, args)
        # Show the result on the original image
        if self.show:
            cv2.imshow('Result', np.array(result))
            cv2.imshow('Masks Only', np.array(black_result))
            cv2.waitKey(1)

    def filter_class(self, masks, boxes, labels):
        USED_CLASS_INDICES =  ['__background__', 
    'divider-line',
    'dotted-line',
    'double-line',
    'random-line',
    'road-sign-line',
    'solid-line']
        filtered_masks = []
        filtered_boxes = []
        filtered_labels = []
        for mask, box, label in zip(masks, boxes, labels):
            if label in USED_CLASS_INDICES:  # 사용할 클래스의 인덱스인지 확인
                filtered_masks.append(mask)
                filtered_boxes.append(box)
                filtered_labels.append(label)
        return filtered_masks, filtered_boxes, filtered_labels

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--threshold', default=0.9, type=float, help='score threshold for discarding detection')
    parser.add_argument('-w', '--weights', default='model/model_15.pth', help='path to the trained weight file')
    parser.add_argument('--show', action='store_true', help='whether to visualize the results in real-time on screen')
    parser.add_argument('--no-boxes', action='store_true', help='do not show bounding boxes, only show segmentation map')
    args = parser.parse_args()

    try:
        node = InferenceNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Lane detection inference node terminated.")
    finally:
        if args.show:
            cv2.destroyAllWindows()
