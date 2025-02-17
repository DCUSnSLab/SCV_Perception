# image_display_window.py

import cv2
from PySide6.QtWidgets import QWidget, QLabel, QGridLayout, QMainWindow
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

class ImageDisplayWindow(QMainWindow):
    def __init__(self, aroundView, front_bev_img, back_bev_img, back_undistort_img):
        super().__init__()

        self.setWindowTitle("Image Display")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QGridLayout(central_widget)

        # 이미지를 QLabel로 표시
        label_aroundView = QLabel(self)
        label_aroundView.setPixmap(self.convert_cv_qt(aroundView))
        layout.addWidget(label_aroundView, 0, 0)

        label_front_bev = QLabel(self)
        label_front_bev.setPixmap(self.convert_cv_qt(front_bev_img))
        layout.addWidget(label_front_bev, 0, 1)

        label_back_bev = QLabel(self)
        label_back_bev.setPixmap(self.convert_cv_qt(back_bev_img))
        layout.addWidget(label_back_bev, 1, 0)

        label_back_img = QLabel(self)
        label_back_img.setPixmap(self.convert_cv_qt(back_undistort_img))
        layout.addWidget(label_back_img, 1, 1)

    def convert_cv_qt(self, cv_img):
        """Convert from an OpenCV image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(400, 300, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
