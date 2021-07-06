import re
import skimage
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap, QColor, QIcon
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QFileDialog, QListWidgetItem, QSplashScreen, QApplication
from PyQt5.QtCore import QDir
from Code.process import *
# from setting import setting
from ui_MainWindow import Ui_MainWindow
from Code.image_segmentation import *
from Code.IMDC_extract import *
from Code.RMDC_extract import *
from PyQt5 import QtCore
import os
import numpy as np
import cv2
import sys


class QmyMainWindow(QMainWindow):
    pathChanged = pyqtSignal(str)  # DoubleThreshold传入原图路径
    fileTrans = pyqtSignal(str)  # modify传入原图路径
    labelTrans = pyqtSignal(QImage)  # 传输分割图
    maskTrans = pyqtSignal(QImage)  # 传输遮罩图
    originTrans = pyqtSignal(QImage)  # 修改后的原图
    outdirTrans = pyqtSignal(str)  # 传入输出路径
    initStatus = pyqtSignal(str, str)  # 传入color,transparency

    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_MainWindow()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面

        self.setWindowTitle("焦炭显微图像分析平台")
        self.setWindowIcon(QIcon("images/app.ico"))
        self.showMaximized()
        self.ui.printf('Copyright  2021  by  Wang xialin')

        self.curPixmap = QPixmap()  # 图片
        self.topPixmap = QPixmap()  # 遮罩图片
        self.secondPixmap = QPixmap()
        self.thirdPixmap = QPixmap()
        self.forthPixmap = QPixmap()

        self.scaled = QPixmap()

        self.seg = QImage()  # 分割图
        self.maskimg = QImage()  # 遮罩图
        self.new_origin = QImage()  # 修改后的原图
        self.is_segment_img = False  # 是否分割
        self.is_origin_change = False  # 原图是否修改
        self.is_save_segment = False  # 分割图是否保存
        self.is_save_origin = False  # 修改后的原图是否保存
        self.is_enter_modify = False  # 进入人工标注
        self.is_multi = 0  # 多图模式
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        self.inittuple = self.stuple
        self.pixRatio = 1  # 图像显示比例
        self.zdict = {}  # 文件路径
        self.sdict = {}  # 状态字典
        self.savedict = {}  # 修改原图保存路径
        self.saveseg = {}  # 分割图保存路径
        self.multipath = None

        self.imageName = None
        self.cokeDir = None
        self.cokebatchName = None
        self.resultDir = None

        self.splitNum = 0
        self.predictNum = 0
        self.totalProcessNum = 0
        self.totalNum = 0

        # 输出目录
        self.outdir = None
        self.edit_labelpath = None
        self.is_outdir_change = False
        self.__Flag = (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)  # 节点标志

        self.is_parameters_change = False  # 是否更改参数
        self.input_size = None
        self.overlap_size = None
        self.batch_size = None
        self.gpu_idx = 0
        self.aug_type = 0
        self.mean = None
        self.std = None
        self.use_post_process = 0
        self.unet_path = 'model/weights/background_seg.pkl'
        self.wpunet_path = None
        self.color = None
        self.transparency = None
        # self.pregress = QmyDialog()

        self.origincolor = self.color
        self.qcolor = None

        self.zoomIn = False
        self.zoomOut = False

        self.segmentNum = 0
        self.IMDCNum = 0
        self.RMDCNUm = 0

    # ==============自定义功能函数========================
    def __enableButtons(self):
        """
        工具栏按钮判断
        """
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        if count == 1:
            self.ui.actionImage_Segmentation.setEnabled(True)
            self.ui.actionHE.setEnabled(True)
            self.ui.actionCLAHE.setEnabled(True)
            self.ui.actionUnet.setEnabled(True)
            # self.ui.actionWPUnet.setEnabled(True)
            self.ui.actionDouble_Threshold.setEnabled(True)
            self.ui.actionOTSU.setEnabled(True)

        elif count > 1:
            self.ui.actionNext_Image.setEnabled(True)
            self.ui.actionPrev_Image.setEnabled(True)
            self.ui.actionImage_Segmentation.setEnabled(True)
            self.ui.actionHE.setEnabled(True)
            self.ui.actionCLAHE.setEnabled(True)
            self.ui.actionUnet.setEnabled(True)
            # self.ui.actionWPUnet.setEnabled(True)
            self.ui.actionDouble_Threshold.setEnabled(True)
            self.ui.actionOTSU.setEnabled(True)

        else:
            return

    def __enableSegModify(self):
        """
        工具栏按钮判断
        """
        if self.is_segment_img:
            self.ui.actionHuman_Modify.setEnabled(True)
            self.ui.actionDouble_Column.setEnabled(True)
            self.ui.actionSingle_Column.setEnabled(True)
            self.ui.actionInverse_Value.setEnabled(True)
            self.ui.actionSave.setEnabled(True)
            self.ui.actionSave_Label_as.setEnabled(True)
            self.ui.actionDelete_Label_File.setEnabled(True)
        else:
            self.ui.actionHuman_Modify.setEnabled(False)
            self.ui.actionRMDC.setEnabled(False)
            self.ui.actionDouble_Column.setEnabled(False)
            self.ui.actionSingle_Column.setEnabled(False)
            self.ui.actionInverse_Value.setEnabled(False)
            self.ui.actionSave.setEnabled(False)
            self.ui.actionSave_Label_as.setEnabled(False)
            self.ui.actionDelete_Label_File.setEnabled(False)

    def __enableOriginModify(self):  ##原图相关工具栏按钮判断
        if self.is_origin_change:
            self.ui.actionSave_Origin_as.setEnabled(True)
            self.ui.actionSave.setEnabled(True)
        else:
            self.ui.actionSave_Origin_as.setEnabled(False)
            self.ui.actionSave.setEnabled(False)

    def resetstatus(self, file):  # 重置分割状态
        tup = self.sdict[file]
        self.is_segment_img = tup[0]  # 是否分割
        self.is_origin_change = tup[1]  # 原图是否修改
        self.is_save_segment = tup[2]  # 分割图是否保存
        self.is_save_origin = tup[3]  # 修改后的原图是否保存
        segtext = file.split(".")[0] + "_label.png"
        dirname = self.zdict[file]
        fullFileName = os.path.join(dirname, segtext)  # 带路径文件名
        if os.path.exists(fullFileName):
            self.is_segment_img = True
            self.is_save_segment = True
            self.saveseg[file] = fullFileName
        if not self.is_save_segment:
            self.is_segment_img = False
        if not self.is_save_origin:
            self.is_origin_change = False
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)  # 初始状态
        self.sdict[file] = self.stuple
        self.__enableOriginModify()
        self.__enableSegModify()

    def cleartop(self):  # 清空遮罩
        pix1 = QPixmap(self.topPixmap.size())
        pix1.fill(QtCore.Qt.transparent)
        pix0 = pix1.scaled(self.scaled.size())
        self.topPixmap = pix0
        self.ui.LabPicture.setPixmap(self.topPixmap)

    def clearAll(self):
        """
        清空所有窗口图片
        """
        pix1 = QPixmap(self.secondPixmap.size())
        pix1.fill(QtCore.Qt.transparent)
        pix0 = pix1.scaled(self.scaled.size())
        self.secondPixmap = pix0
        self.ui.LabC.setPixmap(self.secondPixmap)
        self.thirdPixmap = pix0
        self.ui.LabD.setPixmap(self.thirdPixmap)
        self.forthPixmap = pix0
        self.ui.LabE.setPixmap(self.forthPixmap)

    def __getpicPath(self, text):
        """
        图片完整路径
        """
        dirname = self.zdict[text]
        fullFileName = os.path.join(dirname, text)  # 带路径文件名
        return fullFileName

    def __displayPic(self, text):
        """
        显示图片
        """
        fullFileName = self.__getpicPath(text)
        self.curPixmap.load(fullFileName)  # 原始图片
        self.ui.statusBar.showMessage("Loaded" + " " + text, 5000)
        self.on_actZoomFitWin_triggered()  # 适合窗口大小显示

    def is_gray(self, img_path):
        """
        判断是否是灰度图
        """
        try:
            img = skimage.io.imread(img_path)
            if img.ndim == 2:
                return True
            return False

        except Exception as err:
            print("judge gray error:{}".format(err))
            return False

    def is_rgb(self, img_path):
        """
        判断是否是RGB图
        """
        try:
            img = cv2.imread(img_path)
            if img.ndim == 3:
                return True
            return False

        except Exception as err:
            print("judge gray error:{}".format(err))
            return False

    def enhance_img(self, cla):  # 显示修改后的原图
        fname = self.ui.listWidget.currentItem().text()

        cla = cla * 255
        cla1 = cla.astype(np.uint8)
        # print(cla1.dtype)
        shrink = cv2.cvtColor(cla1, cv2.COLOR_BGR2RGB)
        self.new_origin = QImage(shrink.data,
                                 shrink.shape[1],
                                 shrink.shape[0],
                                 QImage.Format_RGB888)
        image3 = self.new_origin

        self.curPixmap = QPixmap.fromImage(image3)
        self.cleartop()
        self.is_origin_change = True
        self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
        self.sdict[fname] = self.stuple
        self.__enableOriginModify()
        pix0 = self.curPixmap.scaled(self.scaled.size())
        self.ui.LabB.setPixmap(pix0)

    # ==============event处理函数==========================
    def wheelEvent(self, event):
        """
        鼠标事件
        """
        angle = event.angleDelta() / 8
        angleY = angle.y()
        # 判断鼠标位置在scrollarea
        labelrect = QRect(self.ui.scrollArea.pos() + self.ui.centralWidget.pos(), self.ui.scrollArea.size())
        if labelrect.contains(event.pos()):
            if angleY > 0:  # 滚轮上滚
                self.on_actZoomIn_triggered()
            else:  # 滚轮下滚
                self.on_actZoomOut_triggered()
        else:
            event.ignore()

    def closeEvent(self, event):
        if self.is_segment_img and not self.is_save_segment and not self.is_enter_modify:
            yes, no = QMessageBox.Yes, QMessageBox.No
            msg = 'Do you want to quit system without saving result?'
            if QMessageBox.warning(self, 'Attention', msg, yes | no) == no:
                event.ignore()
            else:
                return
        else:
            return

    # ==========由connectSlotsByName()自动连接的槽函数============
    @pyqtSlot()
    def on_actZoomFitWin_triggered(self):
        """
        窗口1适应窗口显示图像
        """
        H = self.ui.scrollArea.height()  # scrollArea当前窗口
        realH = self.curPixmap.height()  # 原始图片的实际高度
        pixRatio1 = float(H) / realH  # 当前显示比例，必须转换为浮点数

        W = self.ui.scrollArea.width() - 20
        realW = self.curPixmap.width()
        pixRatio2 = float(W) / realW

        self.pixRatio = max([pixRatio1, pixRatio2])
        pix1 = self.curPixmap.scaled(realW * self.pixRatio, realH * self.pixRatio)
        self.scaled = pix1

        self.ui.LabB.setPixmap(pix1)  # 设置Label的PixMap
        if self.topPixmap is None:
            return
        else:
            pix2 = self.topPixmap.scaled(realW * self.pixRatio, realH * self.pixRatio)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()
    def on_actZoomFitWin_triggered2(self):
        """
        窗口2适应窗口显示图像
        """
        H = self.ui.scrollArea2.height()  # scrollArea当前窗口
        realH = self.secondPixmap.height()  # 原始图片的实际高度
        pixRatio1 = float(H) / realH  # 当前显示比例，必须转换为浮点数

        W = self.ui.scrollArea2.width() - 20
        realW = self.secondPixmap.width()
        pixRatio2 = float(W) / realW

        self.pixRatio = max([pixRatio1, pixRatio2])
        pix1 = self.secondPixmap.scaled(realW * self.pixRatio, realH * self.pixRatio)
        self.scaled = pix1

        self.ui.LabC.setPixmap(pix1)  # 设置Label的PixMap

    @pyqtSlot()
    def on_actZoomFitWin_triggered3(self):
        """
        窗口3适应窗口显示图像
        """
        H = self.ui.scrollArea3.height()  # scrollArea当前窗口
        realH = self.thirdPixmap.height()  # 原始图片的实际高度
        pixRatio1 = float(H) / realH  # 当前显示比例，必须转换为浮点数

        W = self.ui.scrollArea3.width() - 20
        realW = self.thirdPixmap.width()
        pixRatio2 = float(W) / realW

        self.pixRatio = max([pixRatio1, pixRatio2])
        pix1 = self.thirdPixmap.scaled(realW * self.pixRatio, realH * self.pixRatio)
        self.scaled = pix1

        self.ui.LabD.setPixmap(pix1)  #

    @pyqtSlot()
    def on_actZoomFitWin_triggered4(self):
        """
        窗口4适应窗口显示图像
        """
        H = self.ui.scrollArea4.height()  # scrollArea当前窗口
        realH = self.forthPixmap.height()  # 原始图片的实际高度
        pixRatio1 = float(H) / realH  # 当前显示比例，必须转换为浮点数

        W = self.ui.scrollArea4.width() - 20
        realW = self.forthPixmap.width()
        pixRatio2 = float(W) / realW

        self.pixRatio = max([pixRatio1, pixRatio2])
        pix1 = self.forthPixmap.scaled(realW * self.pixRatio, realH * self.pixRatio)
        self.scaled = pix1

        self.ui.LabE.setPixmap(pix1)  #

    @pyqtSlot()
    def on_actZoomFitW_triggered(self):
        """
        图像适应窗口宽度显示
        """
        W = self.ui.scrollArea.width() - 20
        realW = self.curPixmap.width()
        self.pixRatio = float(W) / realW
        pix1 = self.curPixmap.scaledToWidth(W - 30)
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)  # 设置Label的PixMap
        if self.topPixmap is None:
            return
        else:
            pix2 = self.topPixmap.scaledToWidth(W - 30)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()
    def on_actZoomRealSize_triggered(self):
        """
        显示图像实际大小
        """
        self.pixRatio = 1  # 恢复显示比例为1
        self.scaled = self.curPixmap
        self.ui.LabB.setPixmap(self.curPixmap)
        if self.topPixmap is None:
            return
        else:
            self.ui.LabPicture.setPixmap(self.topPixmap)

    @pyqtSlot()
    def on_actZoomIn_triggered(self):
        """
        图像放大
        """
        self.pixRatio = self.pixRatio * 1.2
        W = self.pixRatio * self.curPixmap.width()
        H = self.pixRatio * self.curPixmap.height()
        pix1 = self.curPixmap.scaled(W, H)  # 图片缩放到指定高度和宽度，保持长宽比例
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)
        if self.topPixmap == None:
            return
        else:
            W0 = self.pixRatio * self.topPixmap.width()
            H0 = self.pixRatio * self.topPixmap.height()
            pix2 = self.topPixmap.scaled(W0, H0)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()
    def on_actZoomOut_triggered(self):
        """
        图像缩小
        """
        self.pixRatio = self.pixRatio * 0.8
        W = self.pixRatio * self.curPixmap.width()
        H = self.pixRatio * self.curPixmap.height()
        pix1 = self.curPixmap.scaled(W, H)  # 图片缩放到指定高度和宽度，保持长宽比例
        self.scaled = pix1
        self.ui.LabB.setPixmap(pix1)
        if self.topPixmap is None:
            return
        else:
            W0 = self.pixRatio * self.topPixmap.width()
            H0 = self.pixRatio * self.topPixmap.height()
            pix2 = self.topPixmap.scaled(W0, H0)
            self.ui.LabPicture.setPixmap(pix2)

    @pyqtSlot()
    def on_actionOpen_triggered(self):
        """
        打开单张图像
        """
        try:
            fileList, flt = QFileDialog.getOpenFileNames(self, "choose an image", "",
                                                         "Images(*.jpg *.bmp *.jpeg *.png *.tif *.tiff)")
            if len(fileList) < 1:
                return
            else:
                self.cleartop()
                fullFileName = fileList[0]  # 带路径文件名

                self.ui.printf('正在打开图像 {}'.format(fullFileName))
                aItem = QListWidgetItem()
                aItem.setText(fullFileName.split("/")[-1])
                new1 = fullFileName.split("/")[-1]

                self.imageName = fullFileName.split("/")[-1]
                self.cokeDir = "//".join(fullFileName.split("/")[0:-3])
                if new1 in self.zdict:
                    dlgTitle = "Attention"
                    strInfo = "A file with the same name exists in the file list\nplease use different name."
                    QMessageBox.information(self, dlgTitle, strInfo)
                    return
                else:
                    self.zdict[fullFileName.split("/")[-1]] = os.path.dirname(fullFileName).replace('/', '\\')
                    self.savedict[fullFileName.split("/")[-1]] = os.path.dirname(fullFileName).replace('/', '\\')
                    self.sdict[fullFileName.split("/")[-1]] = self.inittuple
                    aItem.setCheckState(QtCore.Qt.Unchecked)
                    Flag = self.__Flag
                    aItem.setFlags(Flag)
                    self.resetstatus(fullFileName.split("/")[-1])
                    self.ui.listWidget.addItem(aItem)
                    self.ui.listWidget.setCurrentItem(aItem)
                    self.curPixmap.load(fullFileName)  # 原始图片
                    self.ui.statusBar.showMessage("Loaded" + " " + fullFileName.split("/")[-1], 5000)
                    self.on_actZoomFitWin_triggered()  # 原图大小显示
                    self.__enableButtons()
                self.ui.printf('图像打开完成！')
                self.clearAll()
        except Exception as e:
            print(e)

    @pyqtSlot()
    def on_actionOpen_Dir_triggered(self):
        """
        打开文件夹
        """
        aList = []
        curDir = QDir.currentPath()
        dirStr = QFileDialog.getExistingDirectory(self, "open directory", curDir, QFileDialog.ShowDirsOnly)
        self.cokeDir = "//".join(dirStr.split("/")[0:-2])
        print(self.cokeDir)
        try:
            if dirStr.strip() == '':
                return
            else:
                self.ui.listWidget.blockSignals(True)
                self.ui.listWidget.clear()
                self.ui.listWidget.blockSignals(False)
                self.cleartop()
                self.zdict.clear()
                self.savedict.clear()
                aList.append(dirStr)

                dirObj = QDir(dirStr)  # QDir对象
                filedir = dirStr.replace('/', '\\')
                strList = dirObj.entryList(QDir.Files)

                self.ui.printf('正在打开文件夹 {}'.format(filedir))
                try:
                    for line in strList:

                        if line.split("_")[-1] != "label.png" and (
                                line.endswith('jpg') or line.endswith('bmp') or line.endswith('jpeg') or line.endswith(
                            'png') or line.endswith('JPG') or line.endswith('BMP') or line.endswith(
                            'JPEG') or line.endswith('PNG') or line.endswith('tif')):
                            if line in self.zdict:
                                dlgTitle = "Attention"
                                strInfo = "A file with the same name exists in the file list\nplease use different name."
                                QMessageBox.information(self, dlgTitle, strInfo)
                                return
                            else:
                                self.zdict[line] = filedir
                                self.savedict[line] = filedir
                                self.sdict[line] = self.inittuple
                                aItem = QListWidgetItem()
                                aItem.setText(line)
                                aItem.setCheckState(QtCore.Qt.Unchecked)
                                Flag = self.__Flag
                                aItem.setFlags(Flag)
                                self.ui.listWidget.addItem(aItem)
                except Exception as e:
                    dlgTitle = "Attention"
                    strInfo = "There are no qualified files in the current directory\nplease check the suffix, " \
                              "the software only support ‘png’, ‘jpg, ‘jpeg’, ‘bmp’ "
                    QMessageBox.about(self, dlgTitle, strInfo)
                    print(e)
                try:
                    self.ui.listWidget.blockSignals(True)
                    self.ui.listWidget.setCurrentRow(0)
                    self.ui.listWidget.blockSignals(False)
                    filetext = self.ui.listWidget.item(0).text()
                    self.resetstatus(filetext)
                    if self.is_save_segment:
                        spath = self.saveseg[filetext]
                        self.qcolor = self.text2rgb(self.color)
                        im = QImage(spath)
                        image2 = im.convertToFormat(QImage.Format_ARGB32)
                        self.seg = im
                        width = im.width()
                        height = im.height()
                        for x in range(width):
                            for y in range(height):
                                if (image2.pixel(x, y) == 0xFF000000):
                                    image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                else:
                                    image2.setPixelColor(x, y, self.qcolor)
                        self.maskimg = image2
                        self.topPixmap = QPixmap.fromImage(image2)
                        self.labelpath = spath

                        pix0 = self.topPixmap.scaled(self.scaled.size())
                        self.ui.LabPicture.setPixmap(pix0)
                        self.clearAll()

                    print(filetext)
                    self.imageName = filetext
                    self.__displayPic(filetext)
                    self.__enableButtons()
                except Exception as e:
                    dlgTitle = "Attention"
                    strInfo = "There are no qualified files in the current directory\nplease check the suffix, " \
                              "the software only support ‘png’, ‘jpg, ‘jpeg’, ‘bmp’ "
                    QMessageBox.information(self, dlgTitle, strInfo)
                    print(e)
        except Exception as e:
            print(e)

    def on_listWidget_currentItemChanged(self, current, previous):
        """
        切换图片
        """
        self.ui.actionImage_Segmentation.setEnabled(True)
        self.ui.actionSave.setEnabled(False)
        self.ui.actionHuman_Modify.setEnabled(False)
        self.ui.actionRMDC.setEnabled(False)

        try:
            filetext = current.text()
            self.imageName = filetext

            if self.ui.actionMulti_Image_Process.isChecked() and self.is_save_segment:
                self.cleartop()
                img_path = self.__getpicPath(filetext)  # 带路径文件名
                self.qcolor = self.text2rgb(self.color)
                self.__displayPic(filetext)
                if self.is_multi == 2:
                    multipath = self.Omultipath
                else:
                    multipath = self.Umultipath
                try:
                    spath = os.path.splitext(img_path)[0] + '_label.png'
                    outdir = multipath.replace('/', '\\')
                    spath = os.path.join(outdir, spath.split("\\")[-1])

                    im = QImage(spath)
                    image2 = im.convertToFormat(QImage.Format_ARGB32)
                    self.seg = im
                    width = im.width()
                    height = im.height()
                    for x in range(width):
                        for y in range(height):
                            if (image2.pixel(x, y) == 0xFF000000):
                                image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                            else:
                                image2.setPixelColor(x, y, self.qcolor)
                    self.maskimg = image2
                    self.topPixmap = QPixmap.fromImage(image2)
                    self.labelpath = spath

                    pix0 = self.topPixmap.scaled(self.scaled.size())
                    self.ui.LabPicture.setPixmap(pix0)
                    self.resetstatus(filetext)
                except Exception as e:
                    print(e)
            else:
                if self.is_segment_img:

                    if self.is_save_segment:
                        self.resetstatus(filetext)
                        self.cleartop()
                        self.__displayPic(filetext)

                        if self.is_save_segment:

                            spath = self.saveseg[filetext]

                            im = QImage(spath)
                            image2 = im.convertToFormat(QImage.Format_ARGB32)
                            self.seg = im
                            width = im.width()
                            height = im.height()
                            self.qcolor = self.text2rgb(self.color)
                            for x in range(width):
                                for y in range(height):
                                    if (image2.pixel(x, y) == 0xFF000000):
                                        image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                    else:
                                        image2.setPixelColor(x, y, self.qcolor)
                            self.maskimg = image2
                            self.topPixmap = QPixmap.fromImage(image2)
                            self.labelpath = spath

                            pix0 = self.topPixmap.scaled(self.scaled.size())
                            self.ui.LabPicture.setPixmap(pix0)

                    else:

                        yes, no = QMessageBox.Yes, QMessageBox.No
                        msg = 'Do you want to leave current image without saving result?'
                        if (QMessageBox.warning(self, 'Attention', msg, yes | no) == yes):

                            self.cleartop()

                            self.__displayPic(filetext)
                            self.resetstatus(filetext)
                            if self.is_save_segment:
                                spath = self.saveseg[filetext]
                                self.qcolor = self.text2rgb(self.color)
                                im = QImage(spath)
                                image2 = im.convertToFormat(QImage.Format_ARGB32)
                                self.seg = im
                                width = im.width()
                                height = im.height()
                                for x in range(width):
                                    for y in range(height):
                                        if (image2.pixel(x, y) == 0xFF000000):
                                            image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                        else:
                                            image2.setPixelColor(x, y, self.qcolor)
                                self.maskimg = image2
                                self.topPixmap = QPixmap.fromImage(image2)
                                self.labelpath = spath

                                pix0 = self.topPixmap.scaled(self.scaled.size())
                                self.ui.LabPicture.setPixmap(pix0)

                        else:
                            self.ui.listWidget.blockSignals(True)
                            if previous:
                                self.ui.listWidget.setCurrentItem(previous)

                            else:
                                self.ui.listWidget.setCurrentRow(0)
                        self.ui.listWidget.blockSignals(False)
                else:

                    self.__displayPic(filetext)

                    self.resetstatus(filetext)
                    if self.is_save_segment:
                        spath = self.saveseg[filetext]
                        self.qcolor = self.text2rgb(self.color)
                        im = QImage(spath)
                        image2 = im.convertToFormat(QImage.Format_ARGB32)
                        self.seg = im
                        width = im.width()
                        height = im.height()
                        for x in range(width):
                            for y in range(height):
                                if (image2.pixel(x, y) == 0xFF000000):
                                    image2.setPixelColor(x, y, QColor(0, 0, 0, 0))
                                else:
                                    image2.setPixelColor(x, y, self.qcolor)
                        self.maskimg = image2
                        self.topPixmap = QPixmap.fromImage(image2)
                        self.labelpath = spath

                        pix0 = self.topPixmap.scaled(self.scaled.size())
                        self.ui.LabPicture.setPixmap(pix0)
        except Exception as e:
            print(e)

    @pyqtSlot()
    def on_actionNext_Image_triggered(self):
        """
        切换上一张图像
        """
        self.clearAll()
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        count = count - 1
        currentRow = self.ui.listWidget.currentRow()

        try:
            if currentRow == count:
                # text = self.ui.listWidget.item(currentRow).text()
                # self.__displayPic(text)
                return

            elif currentRow < count:

                self.ui.listWidget.setCurrentRow(currentRow + 1)
        except Exception as e:
            print(e)
            return

    @pyqtSlot()
    def on_actionPrev_Image_triggered(self):
        """
        切换上一张图片
        """
        self.clearAll()
        currentRow = self.ui.listWidget.currentRow()
        try:
            if currentRow == 0:
                return

            elif currentRow > 0:
                self.ui.listWidget.setCurrentRow(currentRow - 1)
        except Exception as e:
            print(e)
            return

    @pyqtSlot()
    def on_search_editingFinished(self):
        """
        文件夹内搜索文件
        """
        list = []
        first = 0
        count = self.ui.listWidget.count()  # 获取listwidget中条目数
        for i1 in range(count):
            list.append(self.ui.listWidget.item(i1).text())
        word = self.ui.search.text()

        if word.strip() != '':
            for i2 in range(count):
                if word in list[i2]:
                    t1 = self.ui.listWidget.item(first).text()
                    t2 = self.ui.listWidget.item(i2).text()
                    cItem = self.ui.listWidget.item(first)
                    bItem1 = self.ui.listWidget.item(i2)
                    bItem1.setText(t1)
                    cItem.setText(t2)
                    cItem.setCheckState(QtCore.Qt.Checked)
                    first = first + 1
                else:
                    bItem2 = self.ui.listWidget.item(i2)
                    bItem2.setCheckState(QtCore.Qt.Unchecked)
        else:
            for i3 in range(count):
                dItem = self.ui.listWidget.item(i3)
                dItem.setCheckState(QtCore.Qt.Unchecked)

    @pyqtSlot()
    def on_actionUnet_triggered(self):
        """
        焦炭显微光学组织提取事件
        """
        fname = self.ui.listWidget.currentItem().text()
        img_path = self.__getpicPath(fname)
        # self.qcolor = self.text2rgb(self.color)
        pth_address = QDir.currentPath()

        pattern0 = '-'
        pattern1 = '_'
        pattern2 = '.tif'
        match0 = re.search(pattern0, self.imageName)
        o = match0.start()
        match1 = re.search(pattern1, self.imageName)
        s = match1.start()
        match2 = re.search(pattern2, self.imageName)
        e = match2.start()

        print(self.imageName)
        print(self.cokeDir)

        cokeName = self.imageName[:o]
        cokeBatch = self.imageName[s + 1: e]
        cokeName_Batch = cokeName + '_' + cokeBatch

        self.cokebatchName = cokeName_Batch
        self.resultDir = os.path.join(self.cokeDir, "result", self.cokebatchName)
        self.segmentDir = os.path.join(self.resultDir, self.cokebatchName + "_forground_170.tif")

        try:
            self.ImageSegmentationThread = ImageSegmentationThread(self.imageSegmentationSlot, self.segmentFinishSlot)
            self.ImageSegmentationThread.start()
            self.processbar = Processbar()
            self.processbar.show()
            self.processbar.setValue(0)
        except Exception as e:
            QMessageBox.information(self, "Error", str(e))
            print("segment Error1:{0}".format(e))

    @pyqtSlot()  ##Human_Modify
    def on_actionHuman_Modify_triggered(self):
        self.on_actionSingle_Column_triggered()

    @pyqtSlot()
    def on_actionRMDC_triggered(self):
        """
        焦炭变色颗粒分类事件
        """
        self.RMDCDir = os.path.join(self.resultDir, self.cokebatchName + "_activate_total.tif")
        self.txtDir = os.path.join(self.resultDir, self.cokebatchName + "_result.txt")

        try:
            self.RMDCExtractThread = RMDCExtractThread(self.RMDCExtractSlot, self.RMDCFinishSlot)
            self.RMDCExtractThread.start()
            self.processbar = Processbar()
            self.processbar.show()
            self.processbar.setValue(0)
        except Exception as e:
            QMessageBox.information(self, "Error", str(e))
            print("segment Error1:{0}".format(e))

    @pyqtSlot()
    def on_actionImage_Segmentation_triggered(self):
        self.on_actionUnet_triggered()

    @pyqtSlot()
    def on_actionSingle_Column_triggered(self):
        """
        焦炭惰性物识别事件
        """
        self.IMDCDir = os.path.join(self.resultDir, self.cokebatchName + "_visual.tif")

        try:
            self.IMDCExtractThread = IMDCExtractThread(self.IMDCExtractSlot, self.IMDCFinishSlot)
            self.IMDCExtractThread.start()
            self.processbar = Processbar()
            self.processbar.show()
            self.processbar.setValue(0)
        except Exception as e:
            QMessageBox.information(self, "Error", str(e))
            print("segment Error1:{0}".format(e))

    @pyqtSlot()
    def on_actionChange_Output_Dir_triggered(self):
        """
        选择图像输出目录
        """
        curDir = QDir.currentPath()

        self.outdir = QFileDialog.getExistingDirectory(self, "save labels in directory", curDir,
                                                       QFileDialog.ShowDirsOnly)

        if self.outdir.strip() == '':
            return
        else:
            self.ui.statusBar.showMessage("Change output directory.Labels will be saved in" + " " + self.outdir, 5000)
            self.is_outdir_change = True

    # 保存图像
    @pyqtSlot()
    def on_actionSave_Label_as_triggered(self):

        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)
        default_resultfile_name = os.path.splitext(fullname)[0] + '_label.png'
        filters = "Image (*.png)"
        # basename = os.path.splitext(self.resultname)[0]
        # default_resultfile_name = self.resultname  # os.path.join(self.currentPath(),basename + '.jpg')
        if self.is_outdir_change:
            newpath = os.path.join(self.outdir, fname)
            resultfile_name = os.path.splitext(newpath)[0] + '_label.png'
        else:
            resultfile_name = default_resultfile_name
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'Save Label', resultfile_name,
            filters)
        if filename:
            self.labelpath = str(filename)
            self.saveseg[fname] = self.labelpath
            self.seg.save(self.labelpath, "PNG")
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()
    def on_actionSave_Origin_as_triggered(self):
        """
        另存修改后的原图
        """
        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)
        default_resultfile_name = os.path.splitext(fullname)[0] + '_enhance.png'
        filters = "Image (*.png)"
        # basename = os.path.splitext(self.resultname)[0]
        # default_resultfile_name = self.resultname  # os.path.join(self.currentPath(),basename + '.jpg')
        if self.is_outdir_change:
            newpath = os.path.join(self.outdir, fname)
            resultfile_name = os.path.splitext(newpath)[0] + '_enhance.png'
        else:
            resultfile_name = default_resultfile_name
        filename, filetype = QFileDialog.getSaveFileName(
            self, 'Save Origin', resultfile_name, filters)
        if filename:
            self.outpath = str(filename)
            img = Image.fromqimage(self.new_origin)
            image = img.convert('L')
            image.save(self.outpath)
            updir = os.path.split(self.outpath)

            self.savedict[fname] = updir[0].replace('/', '\\')
            del self.savedict[fname]
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##检测原图是否修改和是否分割图并保存
    def on_actionSave_triggered(self):

        fname = self.ui.listWidget.currentItem().text()
        fullname = self.__getpicPath(fname)

        if self.is_origin_change and not self.is_save_origin and not self.is_segment_img:
            default_oringin_name = os.path.splitext(fullname)[0] + '_enhance.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                origin_name = os.path.splitext(newpath)[0] + '_enhance.png'
            else:
                origin_name = default_oringin_name
            img = Image.fromqimage(self.new_origin)
            image = img.convert('L')
            image.save(origin_name)
            updir = os.path.split(origin_name)
            self.savedict[fname] = updir[0].replace('/', '\\')
            del self.savedict[fname]
            self.statusBar().showMessage('Enhanced image is saved in' + " " + origin_name, 4000)
            self.is_save_origin = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        elif self.is_segment_img and not self.is_save_segment and not self.is_origin_change:
            default_seg_name = os.path.splitext(fullname)[0] + '_label.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                seg_name = os.path.splitext(newpath)[0] + '_label.png'
            else:
                seg_name = default_seg_name
            self.labelpath = seg_name
            print(seg_name)
            self.seg.save(seg_name, "PNG")
            self.saveseg[fname] = self.labelpath
            self.statusBar().showMessage('Label is saved in' + " " + seg_name, 4000)
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        elif self.is_segment_img and self.is_origin_change and not self.is_save_segment and not self.is_save_origin:
            default_oringin_name = os.path.splitext(fullname)[0] + '_enhance.png'

            default_seg_name = os.path.splitext(fullname)[0] + '_label.png'
            if self.is_outdir_change:
                newpath = os.path.join(self.outdir, fname)
                origin_name = os.path.splitext(newpath)[0] + '_enhance.png'
                seg_name = os.path.splitext(newpath)[0] + '_label.png'
            else:
                origin_name = default_oringin_name
                seg_name = default_seg_name
            img = Image.fromqimage(self.new_origin)
            image = img.convert('L')
            image.save(origin_name)
            updir = os.path.split(origin_name)
            self.savedict[fname] = updir[0].replace('/', '\\')
            del self.savedict[fname]
            self.labelpath = seg_name
            self.seg.save(seg_name, "PNG")

            self.saveseg[fname] = self.labelpath
            self.statusBar().showMessage(
                'Enhanced image and label are saved separately in' + " " + origin_name + " " + seg_name, 4000)
            self.is_save_origin = True
            self.is_save_segment = True
            self.stuple = (self.is_segment_img, self.is_origin_change, self.is_save_segment, self.is_save_origin)
            self.sdict[fname] = self.stuple
        else:
            return

    @pyqtSlot()  ##参数设置
    def on_actionMethod_Setting_triggered(self):
        self.set.show()

    # =============自定义槽函数===============================
    def imageSegmentationSlot(self, val):
        """
        语义分割函数
        """
        self.segmentNum = int(val)

        self.processbar.setValue(self.segmentNum)
        self.processbar.setText("光学组织提取中...")

    def segmentFinishSlot(self, val):
        """
        语义分割结束
        """
        if val == 1:
            self.processbar.setValue(100)
            self.processbar.close()
            self.secondPixmap.load(self.segmentDir)
            self.on_actZoomFitWin_triggered2()

            self.ui.actionImage_Segmentation.setEnabled(False)
            self.ui.actionSave.setEnabled(True)
            self.ui.actionHuman_Modify.setEnabled(True)

    def IMDCExtractSlot(self, val):
        """
        惰性物识别函数
        """
        self.segmentNum = int(val)

        self.processbar.setValue(self.segmentNum)
        self.processbar.setText("惰性物识别中...")

    def IMDCFinishSlot(self, val):
        """
        惰性物识别结束
        """
        if val == 1:
            self.processbar.setValue(100)
            self.processbar.close()
            self.thirdPixmap.load(self.IMDCDir)
            self.on_actZoomFitWin_triggered3()

            self.ui.actionImage_Segmentation.setEnabled(False)
            self.ui.actionSave.setEnabled(True)
            self.ui.actionHuman_Modify.setEnabled(False)
            self.ui.actionRMDC.setEnabled(True)

    def RMDCExtractSlot(self, val):
        """
        惰性物识别函数
        """
        self.segmentNum = int(val)

        self.processbar.setValue(self.segmentNum)
        self.processbar.setText("变色颗粒分类中...")

    def RMDCFinishSlot(self, val):
        """
        惰性物识别结束
        """
        if val == 1:
            self.processbar.setValue(100)
            self.processbar.close()
            self.forthPixmap.load(self.RMDCDir)
            self.on_actZoomFitWin_triggered4()

            with open(self.txtDir, "r") as f:
                data = f.read()
                self.ui.printf(data)

            self.ui.actionImage_Segmentation.setEnabled(False)
            self.ui.actionSave.setEnabled(True)
            self.ui.actionHuman_Modify.setEnabled(False)
            self.ui.actionRMDC.setEnabled(False)


if __name__ == "__main__":
    # 创建GUI应用程序
    app = QApplication(sys.argv)

    # 显示启动画面Logo
    splash = QSplashScreen(QPixmap("images/hello.png"))
    splash.show()

    # 创建窗体
    form = QmyMainWindow()
    form.show()
    splash.finish(form)
    sys.exit(app.exec_())
