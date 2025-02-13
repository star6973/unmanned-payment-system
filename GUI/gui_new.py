{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/pyo/GUI\n"
     ]
    }
   ],
   "source": [
    "cd GUI"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/pyo/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/serialization.py:593: SourceChangeWarning: source code of class 'mmdet.models.dense_heads.rpn_head.RPNHead' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.\n",
      "  warnings.warn(msg, SourceChangeWarning)\n",
      "/home/pyo/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/serialization.py:593: SourceChangeWarning: source code of class 'mmdet.models.roi_heads.dynamic_roi_head.DynamicRoIHead' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.\n",
      "  warnings.warn(msg, SourceChangeWarning)\n",
      "/home/pyo/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/torch/serialization.py:593: SourceChangeWarning: source code of class 'mmcv.ops.roi_align.RoIAlign' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.\n",
      "  warnings.warn(msg, SourceChangeWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/home/pyo/GUI\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import sys\n",
    "import joblib\n",
    "from PIL import ImageOps\n",
    "from face_recognition import preprocessing\n",
    "from inference.util import draw_bb_on_img\n",
    "from sys import exit\n",
    "import cv2\n",
    "import PyQt5\n",
    "from PyQt5.QtCore import QTimer\n",
    "from PyQt5.QtGui import QImage\n",
    "from PyQt5.QtGui import QPixmap\n",
    "from PyQt5 import QtGui, QtCore\n",
    "from PyQt5.QtWidgets import *\n",
    "from PyQt5.QtCore import *\n",
    "from PyQt5.QtGui import *\n",
    "from PyQt5.QtWidgets import QApplication\n",
    "from PyQt5.QtWidgets import QDialog\n",
    "from PyQt5.uic import loadUi\n",
    "from functools import partial\n",
    "from PIL.ImageQt import ImageQt\n",
    "from PIL import Image\n",
    "import numpy as np\n",
    "import time\n",
    "from mmdet.apis import inference_detector, init_detector\n",
    "from mmcv.utils import is_str\n",
    "import random\n",
    "import mmcv\n",
    "from PIL import Image\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import skimage\n",
    "from skimage import io\n",
    "import torchvision\n",
    "import eval_widerface\n",
    "import torchvision_model\n",
    "import model\n",
    "\n",
    "\n",
    "model = torch.load('save_model/DynamicRCNN_model.pt')\n",
    "score_thr = 0.7\n",
    "CLASSES = ('ID_gum', 'buttering', 'couque_coffee', 'chocopie', 'cidar', \n",
    "           'couque_white', 'coke', 'diget_ori', 'diget_choco', 'gumi_gumi', \n",
    "           'homerunball', 'jjolbyung_noodle', 'juicyfresh', 'jjolbyung_ori', \n",
    "           'spearmint', 'squid_peanut', 'samdasu', 'tuna', 'toreta', \n",
    "           'vita500', 'welchs', 'zec')\n",
    "COST = (990, 1290, 990, 990, 800,\n",
    "        990, 1100, 1990, 2390, 1090,\n",
    "        1390, 990, 990,990,\n",
    "        990, 2690, 690, 2990, 1490,\n",
    "        690, 990, 1490)\n",
    "class_by_color = [[128, 0, 0], [255,0,0], [255,165,0], [128,128,0], [124,252,0], [0,128,0], \n",
    "                  [47,79,79], [0,255,255], [0,206,209], [100,149,237], [25,25,112], [0,0,255], \n",
    "                  [147,112,219], [216,191,216], [199,21,133], [255,105,180], [250,235,215], [160,82,45], \n",
    "                  [205,133,63], [255,222,173], [112,128,144], [0,0,0]]\n",
    "label_list = []\n",
    "label_length = 0\n",
    "stop_flag = False\n",
    "label_dict = {}\n",
    "\n",
    "label_dict_prev = {}\n",
    "model_path = 'out/model.pt'\n",
    "depth = 50\n",
    "scale = 1.0\n",
    "gl_name = \"\"\n",
    "gl_cost = 0\n",
    "capture_1 = 0\n",
    "finish_button_flag = 0\n",
    "\n",
    "print(os.getcwd())\n",
    "\n",
    "def random_color(label_num):\n",
    "    return tuple(class_by_color[label_num])\n",
    "\n",
    "def imshow_det_bboxes(img,\n",
    "                      bboxes,\n",
    "                      labels,\n",
    "                      class_names=None,\n",
    "                      score_thr=0,\n",
    "                      bbox_color='red',\n",
    "                      text_color='red',\n",
    "                      thickness=1.5,\n",
    "                      font_scale=1.5,\n",
    "                      show=True,\n",
    "                      win_name='',\n",
    "                      wait_time=0,\n",
    "                      out_file=None):\n",
    "\n",
    "    assert bboxes.ndim == 2\n",
    "    assert labels.ndim == 1\n",
    "    assert bboxes.shape[0] == labels.shape[0]\n",
    "    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5\n",
    "    img = mmcv.imread(img)\n",
    "\n",
    "    if score_thr > 0:\n",
    "        assert bboxes.shape[1] == 5\n",
    "        scores = bboxes[:, -1]\n",
    "        inds = scores > score_thr\n",
    "        bboxes = bboxes[inds, :]\n",
    "        labels = labels[inds]\n",
    "\n",
    "    img = np.ascontiguousarray(img)\n",
    "    origin_labels = labels\n",
    "    i = 0\n",
    "    for bbox, label in zip(bboxes, labels):\n",
    "        rand_color = random_color(label)\n",
    "        bbox_int = bbox.astype(np.int32)\n",
    "        left_top = (bbox_int[0], bbox_int[1])\n",
    "        right_bottom = (bbox_int[2], bbox_int[3])\n",
    "        cv2.rectangle(img, left_top, right_bottom, rand_color, thickness=thickness)\n",
    "        label_text = class_names[label] if class_names is not None else f'cls {label}'\n",
    "        if len(bbox) > 4:\n",
    "            label_text += f'|{bbox[-1]:.02f}'\n",
    "        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2), cv2.FONT_HERSHEY_COMPLEX, font_scale, rand_color, thickness=2)\n",
    "        i += 1\n",
    "\n",
    "    if show:\n",
    "        imshow(img, win_name, wait_time)\n",
    "    if out_file is not None:\n",
    "        imwrite(img, out_file)\n",
    "    return img, origin_labels\n",
    "\n",
    "def show_result(img,\n",
    "                result,\n",
    "                score_thr=0.3,\n",
    "                bbox_color='magenta',\n",
    "                text_color='magenta',\n",
    "                thickness=3,\n",
    "                font_scale=1.0,\n",
    "                win_name='',\n",
    "                show=False,\n",
    "                wait_time=0,\n",
    "                out_file=None):\n",
    "\n",
    "    img = mmcv.imread(img)\n",
    "    img = img.copy()\n",
    "    if isinstance(result, tuple):\n",
    "        bbox_result, segm_result = result\n",
    "        if isinstance(segm_result, tuple):\n",
    "            segm_result = segm_result[0]  # ms rcnn\n",
    "    else:\n",
    "        bbox_result, segm_result = result, None\n",
    "    bboxes = np.vstack(bbox_result)\n",
    "    labels = [\n",
    "        np.full(bbox.shape[0], i, dtype=np.int32)\n",
    "        for i, bbox in enumerate(bbox_result)\n",
    "    ]\n",
    "    labels = np.concatenate(labels)\n",
    "\n",
    "    if segm_result is not None and len(labels) > 0: # non empty\n",
    "        segms = mmcv.concat_list(segm_result)\n",
    "        inds = np.where(bboxes[:, -1] > score_thr)[0]\n",
    "        np.random.seed(42)\n",
    "        color_masks = [\n",
    "            np.random.randint(0, 256, (1, 3), dtype=np.uint8)\n",
    "            for _ in range(max(labels) + 1)\n",
    "        ]\n",
    "        for i in inds:\n",
    "            i = int(i)\n",
    "            color_mask = color_masks[labels[i]]\n",
    "            mask = segms[i]\n",
    "            img[mask] = img[mask] * 0.5 + color_mask * 0.5\n",
    "\n",
    "    if out_file is not None:\n",
    "        show = False\n",
    "\n",
    "    img, detect_labels = imshow_det_bboxes(img,\n",
    "                                           bboxes,\n",
    "                                           labels,\n",
    "                                           class_names=CLASSES,\n",
    "                                           score_thr=score_thr,\n",
    "                                           bbox_color=bbox_color,\n",
    "                                           text_color=text_color,\n",
    "                                           thickness=thickness,\n",
    "                                           font_scale=font_scale,\n",
    "                                           win_name=win_name,\n",
    "                                           show=show,\n",
    "                                           wait_time=wait_time,\n",
    "                                           out_file=out_file)\n",
    "\n",
    "    if not (show or out_file):\n",
    "        return img, detect_labels\n",
    "\n",
    "#sseven\n",
    "class Yesbutton_class(QDialog):    \n",
    "    def __init__(self,parent):\n",
    "        super(Yesbutton_class, self).__init__(parent)\n",
    "        Yesbutton_ui = 'fiveth(payment).ui'\n",
    "        loadUi(Yesbutton_ui, self)\n",
    "\n",
    "        \n",
    "        self.show()\n",
    "        \n",
    "        self.pay_timer = QTimer(self)\n",
    "        self.pay_timer.timeout.connect(self.pay_window_close)\n",
    "        self.pay_timer.start(3000)\n",
    "        \n",
    "    def pay_window_close(self):\n",
    "        self.close()\n",
    "        \n",
    "        \n",
    "        \n",
    "# 결제 화면(5)\n",
    "class PaymentWindow(QDialog):    \n",
    "    def __init__(self, parent):\n",
    "        super(PaymentWindow, self).__init__(parent)\n",
    "        \n",
    "        global label_dict\n",
    "        global gl_cost\n",
    "        global capture_1\n",
    "        \n",
    "      #  payment_ui = '/home/jmh/mmdetection/GUI/fiveth(payment).ui'\n",
    "        payment_ui = 'seven.ui'\n",
    "        loadUi(payment_ui, self)\n",
    "        self.productlist.setColumnCount(3)\n",
    "        column_headers = ['품목', '수량', '가격']\n",
    "        self.productlist.setHorizontalHeaderLabels(column_headers)\n",
    "         \n",
    "                   \n",
    "        \n",
    "        \n",
    "#         label = self.product_cap\n",
    "#         cap = ImageQt(capture_1)\n",
    "#         cap1 = QtGui.QPixmap.fromImage(cap)\n",
    "#         self.product_cap.setPixmap(cap1)\n",
    "#         self.product_cap.setScaledContents(True)\n",
    "\n",
    "        qformat=QImage.Format_Indexed8\n",
    "        if len(capture_1.shape)==3:\n",
    "            if capture_1.shape[2]==4:\n",
    "                qformat=QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat=QImage.Format_RGB888\n",
    "\n",
    "        outImage=QImage(capture_1,capture_1.shape[1],capture_1.shape[0],capture_1.strides[0],qformat)\n",
    "        outImage=outImage.rgbSwapped()\n",
    "        \n",
    "\n",
    "        \n",
    "        self.product_cap.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.product_cap.setScaledContents(True)\n",
    "        \n",
    "        \n",
    "            \n",
    "            \n",
    "            \n",
    "        self.receipt()\n",
    "        self.productlist_func()\n",
    "        self.Yesbutton.clicked.connect(self.Yesbutton_1)\n",
    "        self.Nobutton.clicked.connect(self.Nobutton_1)\n",
    "        self.show()\n",
    "        \n",
    "    def Yesbutton_1(self):\n",
    "        Yesbutton_class(self)\n",
    "\n",
    "    def Nobutton_1(self):\n",
    "        global stop_flag\n",
    "        stop_flag = False\n",
    "        self.close()\n",
    "\n",
    "        \n",
    "    def productlist_func(self):\n",
    "        result = 0\n",
    "        self.productlist.clearContents()\n",
    "        \n",
    "        rowcount = len(list(label_dict.keys()))\n",
    "        self.productlist.setRowCount(rowcount)\n",
    "        for ld in range(len(label_dict.keys())):\n",
    "                col1 = QTableWidgetItem(str(list(label_dict.keys())[ld]))\n",
    "                self.productlist.setItem(ld,0,col1)\n",
    "\n",
    "                col2 = QTableWidgetItem(str(list(label_dict.values())[ld]))\n",
    "                self.productlist.setItem(ld,1,col2)\n",
    "\n",
    "                cost = list(label_dict.values())[ld] * COST[ld]\n",
    "                result += cost\n",
    "                col3 = QTableWidgetItem(str(cost))\n",
    "                self.productlist.setItem(ld,2,col3)\n",
    "\n",
    "                self.Totalcost.clear()\n",
    "                self.Totalcost.addItem(\"     \" + str(gl_cost)+\"  Won\")\n",
    "            \n",
    "    def receipt(self):\n",
    "        global gl_name\n",
    "        self.nameLabel.addItem(gl_name)\n",
    "\n",
    "\n",
    "# 객체 인식 화면(4)\n",
    "class ObjectDetection(QDialog):\n",
    "    def __init__(self, parent):\n",
    "        super(ObjectDetection, self).__init__(parent)\n",
    "       # detection_ui = '/home/jmh/mmdetection/GUI/fourth(detection).ui'\n",
    "        detection_ui = 'fourth(detection).ui'\n",
    "        loadUi(detection_ui, self)\n",
    "        self.image = None\n",
    "        self.processedImage = None\n",
    "        \n",
    "        self.start_webcam()\n",
    "        #self.detectButton.setCheckable(True)\n",
    "        \n",
    "        self.stopButton.clicked.connect(self.stop_webcam)\n",
    "        self.face_Enabled = True\n",
    "        self.tableWidget.setColumnCount(3)\n",
    "        column_headers = ['품목', '수량', '가격']\n",
    "        self.tableWidget.setHorizontalHeaderLabels(column_headers)\n",
    "        self.show()\n",
    "\n",
    "    def goto_payment(self):\n",
    "        PaymentWindow(self)\n",
    "\n",
    "    def start_webcam(self):\n",
    "        self.capture = cv2.VideoCapture(2)\n",
    "        self.capture2 = cv2.VideoCapture(4)\n",
    "       \n",
    "        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)\n",
    "        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)\n",
    "        \n",
    "        self.capture2.set(cv2.CAP_PROP_FRAME_HEIGHT,480)\n",
    "        self.capture2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)\n",
    "        \n",
    "        self.timer = QTimer(self)\n",
    "        self.timer.timeout.connect(self.update_frame)\n",
    "        self.timer.start(1)\n",
    "\n",
    "    def update_frame(self):\n",
    "        global gl_name\n",
    "        gl_name = \"\"\n",
    "        self.flag = 0\n",
    "        ret, self.image = self.capture.read()\n",
    "        ret, self.frame = self.capture2.read()\n",
    "        \n",
    "        self.image = cv2.flip(self.image, 1)\n",
    "        self.frame = cv2.flip(self.frame, 1)\n",
    "        \n",
    "        self.frame = Image.fromarray(self.frame)        \n",
    "        self.displayImage(self.image, 2)\n",
    "        \n",
    "        detected_image, detected_labels = self.detect_product(self.image)\n",
    "        self.detectedImage = detected_image\n",
    "\n",
    "        self.displayImage(detected_image, 2)\n",
    "        self.displayLabels(detected_labels)\n",
    "            \n",
    "        face_recogniser = joblib.load('model/face_recogniser.pkl')\n",
    "        preprocess = preprocessing.ExifOrientationNormalize()\n",
    "        faces = face_recogniser(preprocess(self.frame))\n",
    "        if  faces != []:\n",
    "            draw_bb_on_img(faces, self.frame)\n",
    "            gl_name = faces[0][0][0]\n",
    "            \n",
    "            self.displayFace(self.frame)\n",
    "        \n",
    "    def displayFace(self, img):\n",
    "        pix = np.array(img)\n",
    "        \n",
    "        qformat=QImage.Format_Indexed8\n",
    "        if len(pix.shape)==3:\n",
    "            if pix.shape[2]==4:\n",
    "                qformat=QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat=QImage.Format_RGB888\n",
    "\n",
    "        outImage=QImage(pix,pix.shape[1],pix.shape[0],pix.strides[0],qformat)\n",
    "        outImage=outImage.rgbSwapped()\n",
    "\n",
    "        \n",
    "        \n",
    "        self.imgLabel.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.imgLabel.setScaledContents(True)\n",
    "        \n",
    "    \n",
    "    def detect_product(self, img):\n",
    "        result = inference_detector(model, img)\n",
    "        img, labels = show_result(img, result, score_thr, show=False)\n",
    "        \n",
    "        return img, labels\n",
    "\n",
    "    def stop_webcam(self):\n",
    "        # 결제하기 버튼\n",
    "        global stop_flag\n",
    "        global gl_name\n",
    "        global capture_1\n",
    "        capture_1 = self.detectedImage\n",
    "        \n",
    "        if gl_name:\n",
    "            stop_flag = True\n",
    "            PaymentWindow(self)\n",
    "        else :\n",
    "            pass\n",
    " \n",
    "        \n",
    "    def displayImage(self, img, window=1):\n",
    "        qformat=QImage.Format_Indexed8\n",
    "        if len(img.shape)==3:\n",
    "            if img.shape[2]==4:\n",
    "                qformat=QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat=QImage.Format_RGB888\n",
    "\n",
    "        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)\n",
    "        outImage=outImage.rgbSwapped()\n",
    "\n",
    "        if window==1:\n",
    "            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))\n",
    "            self.imgLabel.setScaledContents(True)\n",
    "\n",
    "        if window==2:\n",
    "            self.processedImgLabel.setPixmap(QPixmap.fromImage(outImage))\n",
    "            self.processedImgLabel.setScaledContents(True)\n",
    "            \n",
    "            \n",
    "    def displayLabels(self, labels):\n",
    "        result = 0\n",
    "        global label_length\n",
    "        global label_dict_prev\n",
    "        global stop_flag\n",
    "        global gl_cost\n",
    "        global label_dict\n",
    "        \n",
    "        if stop_flag == False:\n",
    "            label_dict = {}\n",
    "            for l in labels:\n",
    "                if CLASSES[l] in (label_dict.keys()):\n",
    "\n",
    "                    label_dict[CLASSES[l]] += 1\n",
    "                else:\n",
    "                    label_dict[CLASSES[l]] = 1\n",
    "            if label_dict.keys() == label_dict_prev.keys() and label_dict.values() == label_dict_prev.values():\n",
    "                pass\n",
    "            else:\n",
    "                self.tableWidget.clearContents()\n",
    "                rowcount = len(list(label_dict.keys()))\n",
    "                self.tableWidget.setRowCount(rowcount)\n",
    "                for ld in range(len(label_dict.keys())):\n",
    "                        col1 = QTableWidgetItem(str(list(label_dict.keys())[ld]))\n",
    "                        self.tableWidget.setItem(ld,0,col1)\n",
    "\n",
    "                        col2 = QTableWidgetItem(str(list(label_dict.values())[ld]))\n",
    "                        self.tableWidget.setItem(ld,1,col2)\n",
    "\n",
    "                        cost = list(label_dict.values())[ld] * COST[ld]\n",
    "                        result += cost\n",
    "                        col3 = QTableWidgetItem(str(cost))\n",
    "                        self.tableWidget.setItem(ld,2,col3)\n",
    "\n",
    "                        self.cost_list.clear()\n",
    "                        self.cost_list.addItem(\"     \" + str(result)+\"  Won\")\n",
    "                        gl_cost = result\n",
    "            label_dict_prev = label_dict\n",
    "        else :\n",
    "            pass\n",
    "                \n",
    "\n",
    "        \n",
    "\n",
    "\n",
    "# 얼굴 및 정보 등록 화면(3)\n",
    "# class WaitWindow(QWidget):\n",
    "#     def __init__(self, parent=None):\n",
    "#         super(WaitWindow).__init__()\n",
    "#         loadUi('sixth(wait).ui')\n",
    "#         self.show()\n",
    "\n",
    "class RegisterInfo(QDialog):\n",
    "    def __init__(self, parent=None):\n",
    "        super(RegisterInfo, self).__init__(parent)\n",
    "        #loadUi('/home/jmh/mmdetection/GUI/third(info).ui', self)\n",
    "        loadUi('third(info).ui', self)\n",
    "        \n",
    "        qr = self.frameGeometry()\n",
    "        cp = QDesktopWidget().availableGeometry().center()\n",
    "        qr.moveCenter(cp)\n",
    "        ##############################################################\n",
    "        self.capture = cv2.VideoCapture(4)\n",
    "        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)\n",
    "        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)\n",
    "        \n",
    "\n",
    "        self.timer = QTimer(self)\n",
    "        self.timer.timeout.connect(self.update_frame)\n",
    "        self.timer.start(1)\n",
    "        \n",
    "        self.name = ''\n",
    "        self.dialogs = list()\n",
    "        self.finish_button.clicked.connect(self.next_frame)\n",
    "        self.setWindowTitle('Register Panel')\n",
    "        self.show()\n",
    "\n",
    "    def update_frame(self):\n",
    "        _, self.image = self.capture.read()\n",
    "        self.image = cv2.flip(self.image, 1)\n",
    "        self.displayImage(self.image)\n",
    "        \n",
    "        self.nameButton.clicked.connect(self.get_name)\n",
    "        \n",
    "        self.front_capture_button.clicked.connect(self.capture_front)\n",
    "        self.left_capture_button.clicked.connect(self.capture_left)\n",
    "        self.right_capture_button.clicked.connect(self.capture_right)\n",
    "        self.up_capture_button.clicked.connect(self.capture_up)\n",
    "        self.down_capture_button.clicked.connect(self.capture_down)\n",
    "        \n",
    "        \n",
    "    def get_name(self):\n",
    "        self.name = self.nameLabel.toPlainText()\n",
    "    #    if not(os.path.isdir(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name))):\n",
    "     #       os.makedirs(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name))\n",
    "        if not(os.path.isdir(os.path.join('DB/', self.name))):\n",
    "            os.makedirs(os.path.join('DB/', self.name))\n",
    "    \n",
    "    def capture_front(self):\n",
    "        global finish_button_flag\n",
    "        finish_button_flag += 1\n",
    "        \n",
    "        frame = self.image\n",
    "        qformat = QImage.Format_Indexed8\n",
    "        if len(frame.shape) == 3:\n",
    "            if frame.shape[2] == 4:\n",
    "                qformat = QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat = QImage.Format_RGB888\n",
    "\n",
    "        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)\n",
    "        outImage = outImage.rgbSwapped()\n",
    "#        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_front_img.jpg'))\n",
    "        outImage.save(os.path.join(os.path.join('DB/', self.name), self.name + '_front_img.jpg'))\n",
    "    \n",
    "    \n",
    "        self.front_label.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.front_label.setScaledContents(True)\n",
    "        \n",
    "    def capture_left(self):\n",
    "        global finish_button_flag\n",
    "        finish_button_flag += 1\n",
    "        \n",
    "        frame = self.image\n",
    "        qformat = QImage.Format_Indexed8\n",
    "        if len(frame.shape) == 3:\n",
    "            if frame.shape[2] == 4:\n",
    "                qformat = QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat = QImage.Format_RGB888\n",
    "\n",
    "        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)\n",
    "        outImage = outImage.rgbSwapped()\n",
    "#        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_left_img.jpg'))\n",
    "        outImage.save(os.path.join(os.path.join('DB/', self.name), self.name + '_left_img.jpg'))\n",
    "        \n",
    "\n",
    "        self.left_label.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.left_label.setScaledContents(True)\n",
    "        \n",
    "    def capture_right(self):\n",
    "        global finish_button_flag\n",
    "        finish_button_flag += 1\n",
    "        \n",
    "        frame = self.image\n",
    "        qformat = QImage.Format_Indexed8\n",
    "        if len(frame.shape) == 3:\n",
    "            if frame.shape[2] == 4:\n",
    "                qformat = QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat = QImage.Format_RGB888\n",
    "\n",
    "        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)\n",
    "        outImage = outImage.rgbSwapped()\n",
    "        outImage.save(os.path.join(os.path.join('DB/', self.name), self.name + '_right_img.jpg'))\n",
    "        \n",
    "        self.right_label.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.right_label.setScaledContents(True)\n",
    "        \n",
    "    def capture_up(self):\n",
    "        global finish_button_flag\n",
    "        finish_button_flag += 1\n",
    "        \n",
    "        frame = self.image\n",
    "        qformat = QImage.Format_Indexed8\n",
    "        if len(frame.shape) == 3:\n",
    "            if frame.shape[2] == 4:\n",
    "                qformat = QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat = QImage.Format_RGB888\n",
    "\n",
    "        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)\n",
    "        outImage = outImage.rgbSwapped()\n",
    "        outImage.save(os.path.join(os.path.join('DB/', self.name), self.name + '_up_img.jpg'))\n",
    "        \n",
    "        self.up_label.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.up_label.setScaledContents(True)\n",
    "        \n",
    "        \n",
    "    def capture_down(self):\n",
    "        global finish_button_flag\n",
    "        finish_button_flag += 1\n",
    "        \n",
    "        frame = self.image\n",
    "        qformat = QImage.Format_Indexed8\n",
    "        if len(frame.shape) == 3:\n",
    "            if frame.shape[2] == 4:\n",
    "                qformat = QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat = QImage.Format_RGB888\n",
    "\n",
    "        outImage = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)\n",
    "        outImage = outImage.rgbSwapped()\n",
    "#        outImage.save(os.path.join(os.path.join('/home/jmh/mmdetection/GUI/DB/', self.name), self.name + '_down_img.jpg'))\n",
    "        outImage.save(os.path.join(os.path.join('DB/', self.name), self.name + '_down_img.jpg'))\n",
    "        \n",
    "        self.down_label.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.down_label.setScaledContents(True)\n",
    "    \n",
    "    def next_frame(self):\n",
    "        global finish_button_flag\n",
    "        if finish_button_flag >= 5:\n",
    "            os.system(\"mmdetection/face-recognition/tasks/train.sh DB/\")\n",
    "            os.system(\"python mmdetection/face-recognition/generate_embeddings.py --input-folder DB/ --output-folder out/\")\n",
    "            os.system(\"python mmdetection/face-recognition/train.py -d DB/ -e out/embeddings.txt -l out/labels.txt -c out/class_to_idx.pkl\")\n",
    "\n",
    "            self.show()\n",
    "            self.timer.stop()\n",
    "            self.close()\n",
    "            finish_button_flag = 0\n",
    "        \n",
    "        \n",
    "    def displayImage(self, img):\n",
    "        qformat = QImage.Format_Indexed8\n",
    "        if len(img.shape) == 3:\n",
    "            if img.shape[2] == 4:\n",
    "                qformat = QImage.Format_RGBA888\n",
    "            else:\n",
    "                qformat = QImage.Format_RGB888\n",
    "\n",
    "        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)\n",
    "        outImage = outImage.rgbSwapped()\n",
    "\n",
    "        self.video_frame.setPixmap(QPixmap.fromImage(outImage))\n",
    "        self.video_frame.setScaledContents(True)\n",
    "\n",
    "\n",
    "# 얼굴 등록 or 결제 화면(2)\n",
    "class RegisterWindow(QDialog):\n",
    "    def __init__(self, parent):\n",
    "        super(RegisterWindow, self).__init__(parent)\n",
    "#        register_ui = '/home/jmh/mmdetection/GUI/second(register).ui'\n",
    "        register_ui = 'second(register).ui'\n",
    "        loadUi(register_ui, self)\n",
    "\n",
    "        self.register_button.setFont(QFont(\"12롯데마트드림Bold.ttf\", 15))\n",
    "        self.payment_button.setFont(QFont(\"12롯데마트드림Bold.ttf\", 15))\n",
    "        \n",
    "        self.register_button.clicked.connect(self.faceRegisterButtonFunction)\n",
    "        self.payment_button.clicked.connect(self.paymentButtonFunction)\n",
    "        self.show()\n",
    "\n",
    "    def paymentButtonFunction(self):\n",
    "        ObjectDetection(self)\n",
    "\n",
    "    def faceRegisterButtonFunction(self):\n",
    "        RegisterInfo(self)\n",
    "\n",
    "# 시작 화면(1)\n",
    "class MainWindow(QMainWindow) :\n",
    "    def __init__(self) :\n",
    "        super().__init__()\n",
    "        main_ui = 'first(main).ui'\n",
    "        loadUi(main_ui, self)\n",
    "        self.showMaximized()\n",
    "\n",
    "#        self.title_label.setFont(QFont(\"/home/jmh/mmdetection/GUI/12롯데마트드림Bold.ttf\", 30))\n",
    "#        self.start_button.setFont(QFont(\"/home/jmh/mmdetection/GUI/12롯데마트드림Bold.ttf\", 15))\n",
    "\n",
    "        self.title_label.setFont(QFont(\"12롯데마트드림Bold.ttf\", 30))\n",
    "        self.start_button.setFont(QFont(\"12롯데마트드림Bold.ttf\", 15))\n",
    "\n",
    "        # START 버튼 클릭 시 이벤트 발생\n",
    "        self.start_button.clicked.connect(self.startButtonFunction)\n",
    "\n",
    "    def startButtonFunction(self):\n",
    "        RegisterWindow(self)\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\" :\n",
    "    app = QApplication(sys.argv)\n",
    "    mainWindow = MainWindow()\n",
    "    mainWindow.show()\n",
    "    app.exec_()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
