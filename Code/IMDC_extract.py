# -*- coding: utf-8 -*-
# @Time : 2021/7/6 16:36
# @Author : Xialin.Wang
# @File : IMDC_extract.py
from PyQt5.QtCore import *
import time


class IMDCExtractThread(QThread):
    IMDCProcessSignal = pyqtSignal(int)
    IMDCFinishSignal = pyqtSignal(int)

    def __init__(self, IMDCExtractSlot, IMDCFinishSlot):
        super(IMDCExtractThread, self).__init__()
        self.flag = 1
        self.count = 0
        self.IMDCExtractSlot = IMDCExtractSlot
        self.IMDCFinishSlot = IMDCFinishSlot

        self.IMDCProcessSignal.connect(IMDCExtractSlot)
        self.IMDCFinishSignal.connect(IMDCFinishSlot)

    def run(self):
        try:
            if self.flag != 1:
                return
            else:
                while self.count <= 100:
                    self.count += 1
                    time.sleep(0.01)
                    self.IMDCProcessSignal.emit(self.count)

                self.IMDCFinishSignal.emit(1)

        except RuntimeError as e:
            self.stop()
        except Exception as e:
            self.stop()

    def stop(self):
        self.flag = 0