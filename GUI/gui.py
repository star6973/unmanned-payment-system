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
from mmcv.utils import is_str
import random
import mmcv

model = torch.load('../../content/Object_Detection/mmdetection_object_detection_demo/save_model/SSD_model.pt')
score_thr = 0.5
CLASSES = ('ID_gum', 'buttering', 'couque_coffee', 'chocopie', 'cidar', 
           'couque_white', 'coke', 'diget_ori', 'diget_choco', 'gumi_gumi', 
           'homerunball', 'jjolbyung_noodle', 'juicyfresh', 'jjolbyung_ori', 
           'spearmint', 'squid_peanut', 'samdasu', 'tuna', 'toreta', 
           'vita500', 'welchs', 'zec')
class_by_color = [[128, 0, 0], [255,0,0], [255,165,0], [128,128,0], [124,252,0], [0,128,0], 
                  [47,79,79], [0,255,255], [0,206,209], [100,149,237], [25,25,112], [0,0,255], 
                  [147,112,219], [216,191,216], [199,21,133], [255,105,180], [250,235,215], [160,82,45], 
                  [205,133,63], [255,222,173], [112,128,144], [0,0,0]]
label_list = []

def random_color(label_num):
    return tuple(class_by_color[label_num])

def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='red',
                      text_color='red',
                      thickness=1.5,
                      font_scale=1.5,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None):

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = mmcv.imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    img = np.ascontiguousarray(img)
    origin_labels = labels
    i = 0
    for bbox, label in zip(bboxes, labels):
        rand_color = random_color(label)
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, rand_color, thickness=thickness)
        label_text = class_names[label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2), cv2.FONT_HERSHEY_COMPLEX, font_scale, rand_color, thickness=2)
        i += 1

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img, origin_labels

def show_result(img,
                result,
                score_thr=0.3,
                bbox_color='magenta',
                text_color='magenta',
                thickness=3,
                font_scale=1.0,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):

    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        np.random.seed(42)
        color_masks = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(max(labels) + 1)
        ]
        for i in inds:
            i = int(i)
            color_mask = color_masks[labels[i]]
            mask = segms[i]
            img[mask] = img[mask] * 0.5 + color_mask * 0.5

    if out_file is not None:
        show = False

    img, detect_labels = imshow_det_bboxes(img,
                                           bboxes,
                                           labels,
                                           class_names=CLASSES,
                                           score_thr=score_thr,
                                           bbox_color=bbox_color,
                                           text_color=text_color,
                                           thickness=thickness,
                                           font_scale=font_scale,
                                           win_name=win_name,
                                           show=show,
                                           wait_time=wait_time,
                                           out_file=out_file)

    if not (show or out_file):
        return img, detect_labels

# 객체 인식 화면
class ObjectDetection(QDialog):
    def __init__(self, parent):
        super(ObjectDetection, self).__init__(parent)
        detection_ui = 'GUI/face_detection.ui'
        uic.loadUi(detection_ui, self)
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
            detected_image, detected_labels = self.detect_face(self.image)
            self.displayImage(detected_image, 2)
            self.displayLabels(detected_labels)
        else:
            self.displayImage(self.image, 1)
  
    def detect_face(self, img):
        result = inference_detector(model, img)
        img, labels = show_result(img, result, score_thr, show=False)
        return img, labels

    def stop_webcam(self):
        self.timer.stop()

    def displayImage(self, img, window=1):
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
            
    def displayLabels(self, labels):
        global label_list
        for label in labels:
            if label not in label_list:
                self.listWidget.addItem(str(CLASSES[label]))
                label_list.append(label)

# 얼굴 등록 or 결제 화면
class RegisterWindow(QDialog):
    def __init__(self, parent):
        super(RegisterWindow, self).__init__(parent)
        register_ui = 'GUI/register.ui'
        uic.loadUi(register_ui, self)
        self.payment_button.clicked.connect(self.paymentButtonFunction)
        self.show()

    def paymentButtonFunction(self):
        ObjectDetection(self)

# 시작 화면
class MainWindow(QMainWindow) :
    def __init__(self) :
        super().__init__()
        main_ui = 'GUI/main.ui'
        uic.loadUi(main_ui, self)

        # START 버튼 클릭 시 이벤트 발생
        self.start_button.clicked.connect(self.startButtonFunction)

    def startButtonFunction(self):
        RegisterWindow(self)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_() 