import os
import cv2
import random
import math, time
from PyQt5.QtCore import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QWidget
from PyQt5.QtGui import QImage
import numpy as np
from skimage import io
from skimage import morphology
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr
from Code.dataloader import *
from model.nets.unet_cbam import UnetCBAM


class CokeExtractThread(QThread):
    runtimeSignal = pyqtSignal(int)
    splitProcessSignal = pyqtSignal(int)
    totalNumSignal = pyqtSignal(int)
    predictProcessSignal = pyqtSignal(int)
    totalProcessSignal = pyqtSignal(int)
    extractFinishSignal = pyqtSignal(int)

    def __init__(self, dir, srcImage, splitProcessSlot, totalNumSlot, predictProcessSlot, totalProcessSlot, extractFinishSlot):
        super(CokeExtractThread, self).__init__()
        self.dir = dir
        self.srcImage = srcImage
        self.flag = 1
        self.extractinfoSlot = splitProcessSlot
        self.totalNumSlot = totalNumSlot
        self.predictProcessSlot = predictProcessSlot
        self.totalProcessSlot = totalProcessSlot

        self.totalNumSignal.connect(totalNumSlot)
        self.splitProcessSignal.connect(splitProcessSlot)
        self.predictProcessSignal.connect(predictProcessSlot)
        self.totalProcessSignal.connect(totalProcessSlot)
        self.extractFinishSignal.connect(extractFinishSlot)

    def run(self):
        try:
            if self.flag != 1:
                return
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = self.PytorchModel()
                model.to(device)
                model.eval()

                image_fore_seg_dir = os.path.join(self.dir, 'temp')
                source_dir = os.path.join(image_fore_seg_dir, 'image')
                predict_dir = os.path.join(image_fore_seg_dir, 'predict')
                if not os.path.exists(image_fore_seg_dir):
                    os.mkdir(image_fore_seg_dir)
                    os.mkdir(source_dir)
                    os.mkdir(predict_dir)

                predictImage = self.srcImage
                exampleImage_10 = np.zeros((predictImage.shape[0], predictImage.shape[1], 3), np.uint8)

                self.PytorchSplit(inputImage=predictImage, dir=source_dir, windowSize=700, dstSize=768)
                self.PytorchPredict(model=model, dstDir=source_dir, predictDir=predict_dir, device=device)
                cokeMask = self.PytorchTotal(predictDir=predict_dir, exampleImage=exampleImage_10, windowSize=700, dstSize=768)
                cokeImage = cv2.bitwise_and(predictImage, cv2.cvtColor(cokeMask, cv2.COLOR_GRAY2BGR))
                cv2.imwrite(os.path.join('hehe.tif'), cokeImage)

                # 清空分割过程文件夹
                shutil.rmtree(image_fore_seg_dir)
                self.extractFinishSignal.emit(1)

        except RuntimeError as e:
            self.runtimeSignal.emit()
            self.stop()
            print(e)
        except Exception as e:
            self.runtimeSignal.emit()
            self.stop()
            print(e)

    def stop(self):
        self.flag = 0

    def GenerateLabel(self, inputImage, windowSize):
        weight = inputImage.shape[1]
        height = inputImage.shape[0]

        xSplitPt = np.arange(0, weight - 1, windowSize, np.uint64)
        if weight - xSplitPt[-1] + 1 < windowSize / 3:
            np.delete(xSplitPt[-1])
        xNum = len(xSplitPt)

        ySplitPt = np.arange(0, height - 1, windowSize, np.uint64)
        if height - ySplitPt[-1] + 1 < windowSize / 3:
            np.delete(ySplitPt[-1])
        yNum = len(ySplitPt)

        cnt = np.zeros([1, weight], np.uint64)
        cnt[0, xSplitPt] = 1
        indHeight = np.cumsum(cnt)

        cnt = np.zeros([height, 1], np.uint64)
        cnt[ySplitPt, 0] = 1
        indWeight = np.cumsum(cnt, axis=0)
        indWeight = indWeight - 1
        imageLabel = np.tile(indHeight, [height, 1]) + np.tile(indWeight, [1, weight]) * xNum
        return imageLabel, xNum

    def PytorchSplit(self, inputImage, dir, windowSize=400, dstSize=512):
        srcImage = inputImage.copy()
        imageNum = 0
        interpolation = int((dstSize - windowSize) / 2)

        height = ((srcImage.shape[0] // windowSize) + 1) * windowSize
        weight = ((srcImage.shape[1] // windowSize) + 1) * windowSize

        fillImage = np.zeros([height, weight, 3], np.uint8)
        fillImage[:srcImage.shape[0], :srcImage.shape[1]] = srcImage

        imgLabel, xNum = self.GenerateLabel(fillImage, windowSize=windowSize)
        totoalNum = int(imgLabel[-1, -1])

        # 将图像总数发送给槽函数
        self.totalNumSignal.emit(totoalNum)

        for i in range(1, totoalNum + 1):
            # 将图像分块进展发送给槽函数
            self.splitProcessSignal.emit(i+1)

            index = np.argwhere(imgLabel == i)
            x1 = index[0][0]
            y1 = index[0][1]
            x2 = index[-1][0] + 1
            y2 = index[-1][1] + 1

            if x1 == 0 or y1 == 0 or x2 == height or y2 == weight:
                tempImage1 = fillImage[x1:x2, y1:y2]
                dstImage1 = cv2.copyMakeBorder(tempImage1, interpolation, interpolation, interpolation, interpolation,
                                               cv2.BORDER_DEFAULT)
                # cv2.imwrite('data/test/image' + '/' + str(imageNum).zfill(4) + '.tif', dstImage1)
                cv2.imwrite(os.path.join(dir, str(imageNum).zfill(4) + '.tif'), dstImage1)
            else:
                tempImage2 = fillImage[x1 - interpolation:x2 + interpolation, y1 - interpolation:y2 + interpolation]
                # cv2.imwrite('data/test/image' + '/' + str(imageNum).zfill(4) + '.tif', tempImage2)
                cv2.imwrite(os.path.join(dir, str(imageNum).zfill(4) + '.tif'), tempImage2)
            imageNum = imageNum + 1
        return imageNum

    def PytorchTotal(self, predictDir, exampleImage, windowSize=400, dstSize=512):
        imageNum = 0
        interpolation = int((dstSize - windowSize) / 2)

        inputImage1 = exampleImage
        height = ((inputImage1.shape[0] // windowSize) + 1) * windowSize
        weight = ((inputImage1.shape[1] // windowSize) + 1) * windowSize

        fillImage1 = np.zeros([height, weight], np.uint8)
        imgLabel, xNum = self.GenerateLabel(fillImage1, windowSize=windowSize)
        totoalNum = int(imgLabel[-1, -1])

        for i in range(1, totoalNum + 1):
            # 将图像拼接进展发送给槽函数
            self.totalProcessSignal.emit(i+1)

            srcImage = cv2.cvtColor(cv2.imread(os.path.join(predictDir, str(imageNum).zfill(4) + '.tif')),
                                    cv2.COLOR_BGR2GRAY)

            index = np.argwhere(imgLabel == i)
            x1 = index[0][0]
            y1 = index[0][1]
            x2 = index[-1][0] + 1
            y2 = index[-1][1] + 1
            fillImage1[x1:x2, y1:y2] = srcImage[interpolation:dstSize - interpolation,
                                       interpolation:dstSize - interpolation]
            imageNum = imageNum + 1

        fillImage1 = fillImage1[:inputImage1.shape[0], :inputImage1.shape[1]]
        return fillImage1

    def PytorchModel(self):
        model = UnetCBAM(in_ch=3, out_ch=1, filterNum=16)
        state_dict_load = torch.load('model/weights/coke_system.pkl')
        model.load_state_dict(state_dict_load)

        return model

    def PytorchPredict(self, model, dstDir, predictDir, device):
        imageDir = dstDir
        testLoader = TestDataset(imageDir=imageDir, imageSize=768)

        for i in range(testLoader.size):
            # 将图像推理进展发送给槽函数
            self.predictProcessSignal.emit(i+1)

            image, name = testLoader.load_data()
            image = image.to(device)

            outputs = model(image)
            outputs = (outputs.cpu().data.numpy().squeeze() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(predictDir, name), outputs)



