import os
import sys
from sys import exit
import cv2
import PyQt5
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from functools import partial
from PIL.ImageQt import ImageQt
from PIL import Image
import numpy as np
import time
from mmdet.apis import inference_detector, init_detector
from mmcv.utils import is_str
import random
import mmcv

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
from skimage import io
import torchvision
import eval_widerface
import torchvision_model
import model

model = torch.load('/home/jmh/mmdetection/save_model/DynamicRCNN_model.pt')
score_thr = 0.7
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

model_path = 'out/model.pt'
depth = 50
scale = 1.0

print(os.getcwd())

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

    if segm_result is not None and len(labels) > 0: # non empty
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


# 결제 화면(5)
class PaymentWindow(QDialog):
    def __init__(self, parent):
        super(PaymentWindow, self).__init__(parent)
        payment_ui = '/home/jmh/mmdetection/GUI/fiveth(payment).ui'
        loadUi(payment_ui, self)
        self.show()


# 객체 인식 화면(4)
class ObjectDetection(QDialog):
    def __init__(self, parent):
        super(ObjectDetection, self).__init__(parent)
        detection_ui = '/home/jmh/mmdetection/GUI/fourth(detection).ui'
        loadUi(detection_ui, self)
        self.image = None
        self.processedImage = None
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)
        self.detectButton.setCheckable(True)
        self.detectButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled = False
        self.payButton.clicked.connect(self.goto_payment)
        self.show()

    def goto_payment(self):
        PaymentWindow(self)


    def detect_webcam_face(self, status):
        if status:
            self.detectButton.setText('Stop Detection')
            self.face_Enabled = True
        else:
            self.detectButton.setText('Detect Face')
            self.face_Enabled = False

    def start_webcam(self):
        self.capture = cv2.VideoCapture(2)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image=cv2.flip(self.image, 1)

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
                # self.listWidget.addItem(str(CLASSES[label]))
                label_list.append(label)


# 얼굴 및 정보 등록 화면(3)
# class WaitWindow(QWidget):
#     def __init__(self, parent=None):
#         super(WaitWindow).__init__()
#         loadUi('sixth(wait).ui')
#         self.show()

class RegisterInfo(QDialog):
    def __init__(self, parent=None):
        super(RegisterInfo, self).__init__(parent)
        loadUi('/home/jmh/mmdetection/GUI/third(info).ui', self)
        
        self.capture = cv2.VideoCapture(-1)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(3)
        
        self.name = ''
        self.dialogs = list()
        
        self.setWindowTitle('Register Panel')
        self.show()

    def update_frame(self):
        _, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.displayImage(self.image)
        
        self.nameButton.clicked.connect(self.get_name)
        
        self.front_capture_button.clicked.connect(self.capture_front)
        self.left_capture_button.clicked.connect(self.capture_left)
        self.right_capture_button.clicked.connect(self.capture_right)
        self.up_capture_button.clicked.connect(self.capture_up)
        self.down_capture_button.clicked.connect(self.capture_down)
        
        self.finish_button.clicked.connect(self.next_frame)
        
    def get_name(self):
        self.name = self.nameLabel.toPlainText()
        if not(os.path.isdir(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name))):
            os.makedirs(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name))
    
    def capture_front(self):
        frame = self.image
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_front_img.jpg'))
        
        image = Image.open(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_front_img.jpg'))
        qimage = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.front_label.setPixmap(pixmap)
        self.front_label.setScaledContents(True)
        
    def capture_left(self):
        frame = self.image
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_left_img.jpg'))
        
        image = Image.open(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_left_img.jpg'))
        qimage = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.left_label.setPixmap(pixmap)
        self.left_label.setScaledContents(True)
        
    def capture_right(self):
        frame = self.image
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_right_img.jpg'))
        
        image = Image.open(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_right_img.jpg'))
        qimage = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.right_label.setPixmap(pixmap)
        self.right_label.setScaledContents(True)
        
    def capture_up(self):
        frame = self.image
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_up_img.jpg'))
        
        image = Image.open(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_up_img.jpg'))
        qimage = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.up_label.setPixmap(pixmap)
        self.up_label.setScaledContents(True)
        
        
    def capture_down(self):
        frame = self.image
        qformat = QImage.Format_Indexed8
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_down_img.jpg'))
        
        image = Image.open(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_down_img.jpg'))
        qimage = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.down_label.setPixmap(pixmap)
        self.down_label.setScaledContents(True)
    
    def next_frame(self):
        self.close()
        
#         dialog = WaitWindow()
        
        os.system("/home/jmh/mmdetection/face-recognition/tasks/train.sh /home/jmh/mmdetection/GUI/DB/")
        os.system("python /home/jmh/mmdetection/face-recognition/generate_embeddings.py --input-folder /home/jmh/mmdetection/GUI/DB/ --output-folder /home/jmh/mmdetection/GUI/out/")
        os.system("python /home/jmh/mmdetection/face-recognition/train.py -d /home/jmh/mmdetection/GUI/DB/ -e /home/jmh/mmdetection/GUI/out/embeddings.txt -l /home/jmh/mmdetection/GUI/out/labels.txt -c /home/jmh/mmdetection/GUI/out/class_to_idx.pkl")
        exit()
        
#         if os.path.isfile('./out/class_to_idx.pkl'):
#             dialog.close()
        
    def displayImage(self, img):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage = outImage.rgbSwapped()

        self.video_frame.setPixmap(QPixmap.fromImage(outImage))
        self.video_frame.setScaledContents(True)


# 얼굴 등록 or 결제 화면(2)
class RegisterWindow(QDialog):
    def __init__(self, parent):
        super(RegisterWindow, self).__init__(parent)
        register_ui = '/home/jmh/mmdetection/GUI/second(register).ui'
        loadUi(register_ui, self)

        self.register_button.setFont(QFont("/home/jmh/mmdetection/GUI/12롯데마트드림Bold.ttf", 15))
        self.payment_button.setFont(QFont("/home/jmh/mmdetection/GUI/12롯데마트드림Bold.ttf", 15))
        
        self.register_button.clicked.connect(self.faceRegisterButtonFunction)
        self.payment_button.clicked.connect(self.paymentButtonFunction)
        
        self.show()

    def paymentButtonFunction(self):
        ObjectDetection(self)

    def faceRegisterButtonFunction(self):
        RegisterInfo(self)

# 시작 화면(1)
class MainWindow(QMainWindow) :
    def __init__(self) :
        super().__init__()
        main_ui = '/home/jmh/mmdetection/GUI/first(main).ui'
        loadUi(main_ui, self)

        self.title_label.setFont(QFont("/home/jmh/mmdetection/GUI/12롯데마트드림Bold.ttf", 30))
        self.start_button.setFont(QFont("/home/jmh/mmdetection/GUI/12롯데마트드림Bold.ttf", 15))

        # START 버튼 클릭 시 이벤트 발생
        self.start_button.clicked.connect(self.startButtonFunction)

    def startButtonFunction(self):
        RegisterWindow(self)


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
