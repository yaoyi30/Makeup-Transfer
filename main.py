import os.path
import sys
import cv2
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from models.facedetection import FaceDetection
from models.faceparsing import FaceParsing
from models.facemakeup import FaceMakeup
from PyQt5 import QtCore, QtGui, QtWidgets
from ui_file.mainwindow import Ui_MainWindow

class Main(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)
        self.showMaximized()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)

        self.OriImgName = ''
        self.RefImgName = ''

        self.OriImage = None
        self.RefImage = None

        self.pushButton.clicked.connect(self.choose_ori_img)
        self.pushButton_3.clicked.connect(self.choose_ref_img)

        self.pushButton_2.clicked.connect(self.clear_ori_img)
        self.pushButton_4.clicked.connect(self.clear_ref_img)

        self.pushButton_5.clicked.connect(self.start_transfer)

        if not os.path.exists('output'):
            os.mkdir('output')


    def choose_ori_img(self):
        self.OriImgName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图像文件", ".",
                                                                "图像文件(*.jpg *.jpeg *.png *.tif)")
        if self.OriImgName != '':
            self.OriImage = cv2.cvtColor(cv2.imread(self.OriImgName), cv2.COLOR_BGR2RGB)
            pixmap = QtGui.QPixmap(self.OriImgName)
            scaled_pixmap = pixmap.scaled(self.label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label.setPixmap(scaled_pixmap)

    def clear_ori_img(self):
        self.label.clear()
        self.OriImgName = ''
        self.OriImage = None

    def choose_ref_img(self):
        self.RefImgName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图像文件", ".",
                                                                "图像文件(*.jpg *.jpeg *.png *.tif)")
        if self.RefImgName != '':
            self.RefImage = cv2.cvtColor(cv2.imread(self.RefImgName), cv2.COLOR_BGR2RGB)
            pixmap = QtGui.QPixmap(self.RefImgName)
            scaled_pixmap = pixmap.scaled(self.label_2.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_2.setPixmap(scaled_pixmap)

    def clear_ref_img(self):
        self.label_2.clear()
        self.RefImgName = ''
        self.RefImage = None

    def start_transfer(self):
        if self.OriImage is not None and self.RefImage is not None:
            self.tft = TransferThread(self.OriImage,self.RefImage)
            self.tft.trigger.connect(self.result_show)
            self.tft.start()
        elif self.OriImage is None:
            QMessageBox.warning(self, '警告', '请选择原始图像！')
        elif self.RefImage is None:
            QMessageBox.warning(self, '警告', '请选择参考图像！')
        else:
            QMessageBox.warning(self, '警告', '请选择图像！')

    def result_show(self,result):
        if result == 'Ori Img None':
            QMessageBox.warning(self, '警告', '原图像中无人脸，请重新选择图像！')
        elif result == 'Ref Img None':
            QMessageBox.warning(self, '警告', '参考图像中无人脸，请重新选择图像！')
        else:
            self.label_3.clear()
            pixmap = QtGui.QPixmap('./output/result.png')
            scaled_pixmap = pixmap.scaled(self.label_3.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
            self.label_3.setPixmap(scaled_pixmap)

class TransferThread(QThread):
    trigger = pyqtSignal(str)

    def __init__(self,OriImage,RefImage):
        super(TransferThread, self).__init__()

        self.OriImage = OriImage.copy()
        self.RefImage = RefImage.copy()

        self.facedetection_model = FaceDetection(path='./weights/facedetection.onnx')
        self.faceparsing_model = FaceParsing(path='./weights/faceparsing.onnx')
        self.facemakeup_model = FaceMakeup(path='./weights/facemakeup.onnx')

    def run(self):

        ori_boxes, ori_labels, ori_probs = self.facedetection_model.inference(self.OriImage)
        ref_boxes, ref_labels, ref_probs = self.facedetection_model.inference(self.RefImage)

        if len(ori_boxes) > 0 and len(ref_boxes) > 0:

            if len(ref_boxes) > 1:
                ref_box = ref_boxes[np.argmax(ref_probs)]
            else:
                ref_box = ref_boxes[0]

            for ori_box, ori_label, ori_prob in zip(ori_boxes, ori_labels, ori_probs):

                ori_box[0] = max(ori_box[0] - 25, 0)
                ori_box[1] = max(ori_box[1] - 25, 0)
                ori_box[2] = min(ori_box[2] + 25, self.OriImage.shape[1])
                ori_box[3] = min(ori_box[3] + 25, self.OriImage.shape[0])

                ori_face = self.OriImage[ori_box[1]:ori_box[3], ori_box[0]:ori_box[2], :]
                ori_height, ori_width, _ = ori_face.shape

                ref_box[0] = max(ref_box[0] - 25, 0)
                ref_box[1] = max(ref_box[1] - 25, 0)
                ref_box[2] = min(ref_box[2] + 25, self.OriImage.shape[1])
                ref_box[3] = min(ref_box[3] + 25, self.OriImage.shape[0])

                ref_face = self.RefImage[ref_box[1]:ref_box[3], ref_box[0]:ref_box[2], :]
                ref_height, ref_width, _ = ref_face.shape

                ori_parse = self.faceparsing_model.inference(ori_face)
                ref_parse = self.faceparsing_model.inference(ref_face)

                result = self.facemakeup_model.inference(ori_face, ori_parse, ref_face, ref_parse)

                ori_parse_height, ori_parse_width = ori_parse.shape

                for y in range(ori_parse_height):
                    for x in range(ori_parse_width):
                        index = ori_parse[y, x]
                        if index > 13 or index in [7, 8, 9]:
                            result[y, x] = ori_face[y, x]

                self.OriImage[ori_box[1]:ori_box[3], ori_box[0]:ori_box[2]] = result

            if len(ref_boxes) != 1:
                self.trigger.emit("error")

            image = Image.fromarray(self.OriImage)

            image.save('./output/result.png')
            self.trigger.emit("finish")

        elif len(ori_boxes) == 0:
            self.trigger.emit("Ori Img None")
        else:
            self.trigger.emit("Ref Img None")

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    ui = Main()
    ui.show()
    sys.exit(app.exec_())