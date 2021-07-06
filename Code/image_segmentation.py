# -*- coding: utf-8 -*-
# @Time : 2021/7/6 15:44
# @Author : Xialin.Wang
# @File : image_segmentation.py
from PyQt5.QtCore import *
import time


class ImageSegmentationThread(QThread):
    segmentProcessSignal = pyqtSignal(int)
    segmentFinishSignal = pyqtSignal(int)

    def __init__(self, imageSegmentationSlot, segmentFinishSlot):
        super(ImageSegmentationThread, self).__init__()
        self.flag = 1
        self.count = 0
        self.imageSegmentationSlot = imageSegmentationSlot
        self.segmentFinishSlot = segmentFinishSlot

        self.segmentProcessSignal.connect(imageSegmentationSlot)
        self.segmentFinishSignal.connect(segmentFinishSlot)

    def run(self):
        try:
            if self.flag != 1:
                return
            else:
                while self.count <= 100:
                    self.count += 1
                    time.sleep(0.01)
                    self.segmentProcessSignal.emit(self.count)

                self.segmentFinishSignal.emit(1)

        except RuntimeError as e:
            self.stop()
        except Exception as e:
            self.stop()

    def stop(self):
        self.flag = 0


