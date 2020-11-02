import sys
import cv2
import urllib.request
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.uic import loadUi
import numpy as np
import time
import torch
from mmdet.apis import inference_detector, init_detector

main_ui = uic.loadUiType("GUI/main.ui")[0]
model = torch.load('../../content/Object_Detection/mmdetection_object_detection_demo/save_model/SSD_model.pt')
score_thr = 0.5
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

class ObjectDetection(QDialog):
    def __init__(self, parent):
        super(ObjectDetection, self).__init__(parent)
        loadUi('GUI/face_detection.ui', self)
        self.image=None
        self.processedImage=None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled=False
        self.show()

    def detect_webcam_face(self,status):
        if status:
            self.detectButton.setText('Stop Detection')
            self.face_Enabled=True
        else:
            self.detectButton.setText('Detect Face')
            self.face_Enabled=False

    def start_webcam(self):
        self.capture=cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)

        self.displayImage(self.image, 1)
        if(self.face_Enabled):
            detected_image=self.detect_face(self.image)
            self.displayImage(detected_image, 2)
        else:
            self.displayImage(self.image, 1)
    
    def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(img, result, score_thr=score_thr, show=False)
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.show()
        
    def detect_face(self,img):
        result = inference_detector(model, img)
        img = model.show_result(img, result, score_thr=score_thr, show=False)
        return img

    def stop_webcam(self):
        self.timer.stop()

    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888

        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()

        if window==1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)

        if window==2:
            self.processedImgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.processedImgLabel.setScaledContents(True)


class RegisterWindow(QDialog):
    def __init__(self, parent):
        super(RegisterWindow, self).__init__(parent)
        register_ui = 'GUI/register.ui'
        uic.loadUi(register_ui, self)
        self.payment_button.clicked.connect(self.paymentButtonFunction)
        self.show()

    def paymentButtonFunction(self):
        ObjectDetection(self)


class MainWindow(QMainWindow, main_ui) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        # START 버튼 클릭 시 이벤트 발생
        self.start_button.clicked.connect(self.startButtonFunction)

    def startButtonFunction(self):
        RegisterWindow(self)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_() 