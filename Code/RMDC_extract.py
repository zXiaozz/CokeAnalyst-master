# -*- coding: utf-8 -*-
# @Time : 2021/7/6 17:08
# @Author : Xialin.Wang
# @File : RMDC_extract.py
from PyQt5.QtCore import *
import time


class RMDCExtractThread(QThread):
    RMDCProcessSignal = pyqtSignal(int)
    RMDCFinishSignal = pyqtSignal(int)

    def __init__(self, RMDCExtractSlot, RMDCFinishSlot):
        super(RMDCExtractThread, self).__init__()
        self.flag = 1
        self.count = 0
        self.RMDCExtractSlot = RMDCExtractSlot
        self.RMDCFinishSlot = RMDCFinishSlot

        self.RMDCProcessSignal.connect(RMDCExtractSlot)
        self.RMDCFinishSignal.connect(RMDCFinishSlot)

    def run(self):
        try:
            if self.flag != 1:
                return
            else:
                while self.count <= 100:
                    self.count += 1
                    time.sleep(0.01)
                    self.RMDCProcessSignal.emit(self.count)

                self.RMDCFinishSignal.emit(1)

        except RuntimeError as e:
            self.stop()
        except Exception as e:
            self.stop()

    def stop(self):
        self.flag = 0