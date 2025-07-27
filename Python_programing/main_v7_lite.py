
#--------------------------------------------------------
import cv2
import sys
import os
import shutil
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import QWidget, QSlider, QLineEdit, QLabel, QPushButton, QScrollArea,QApplication,QHBoxLayout, QVBoxLayout, QMainWindow
#--------------------------------------------
from window.Main_window import Ui_MainWindow
from window.Calib_window import Ui_Calib_Window
from window.Exit_window import Ui_Exit_window
from window.Train_window import Ui_Train_Window
from window.Restart_window import Ui_Restart_window 
#---------------------------------------------
import imutils
import serial
import serial.tools.list_ports
import time
import webbrowser

#-------------------------------------------
import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

#------------------------------------------------
ports = serial.tools.list_ports.comports()

if len(ports)>0:
    for onePort in ports:
        com_name = str(onePort)
        if com_name[0:2] == 'CO':
            if com_name[4] == ' ':
                Arduino_serial = serial.Serial(com_name[0:4],9600)
            if com_name[5] == ' ':
                Arduino_serial = serial.Serial(com_name[0:6],9600)
    com_status = True
else:
    com_status = False

#print(Arduino_serial)

#----------------------------------------------
cam_id = 0
template_image = cv2.imread("./image/display_image/anh_mau_2.jpg")

#----------------------------

nen2 = cv2.imread('./image/display_image/Picture2.png')
nen3 = cv2.imread("./image/display_image/Picture1.png")
cv2.imwrite('./image/display_image/anh_chup.jpg',nen2)
cv2.imwrite('./image/display_image/anh_yolo.jpg',nen2)

nen4 = cv2.imread('./image/display_image/Picture2.png')

#------------------------------------

'''======================================================================='''
#----------------------------------------------		
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #self.Main_window = QtWidgets.QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)

        #self.setWindowFlag(QtCore.Qt.FramelessWindowHint)

        self.uic.Button_start.clicked.connect(self.read_serial)
        self.uic.Button_exit.clicked.connect(self.exit)
        self.uic.Button_reset.clicked.connect(self.reset)
        self.uic.Button_stop.clicked.connect(self.stop)
        self.uic.Button_calib.clicked.connect(self.calib_show)
        self.uic.Button_train.clicked.connect(self.train_show)

        self.uic.phat_hien_loi.setWidgetResizable(True)
        self.uic.phat_hien_loi.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.resetScroll()

        self.uic.message.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.show_image()

        #img11 = cv2.resize(img11, None, fx=0.5, fy=0.5)
        frame12 = self.convert_cv_qt(nen3)
        #frame12.setMinimumSize(QSize(500,400))
        #frame12.setMaximumSize(QSize(600,500))
        self.uic.cam.setPixmap(frame12)

        self.thread = {}
        self.status_cam = False
        self.cam_id = 0

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    #-----------------------------------------------
    def calib_show(self):
        self.Calib_window = QtWidgets.QMainWindow()
        self.uic1 = Ui_Calib_Window()
        self.uic1.setupUi(self.Calib_window)
        self.Calib_window.show()
        self.uic1.Button_calib.clicked.connect(self.calib)
        self.uic1.Button_save_calib.clicked.connect(self.save_calib)
        self.uic1.Button_close_calib.clicked.connect(self.close_calib)
        self.uic1.Button_cam_calib.clicked.connect(self.make_image_calib)
        self.uic1.Button_camid.clicked.connect(self.cam_id_connect)

        self.uic1.text_calib.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.calib_dem = 0
        
        #try:
            #shutil.rmtree('./image/calib_image')
        #except OSError as e:
            #print('')

        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False
            
            nen0 = cv2.imread('./image/display_image/Picture1.png')
            nen0_cv = self.convert_cv_qt(nen0)
            self.uic.cam.setPixmap(nen0_cv)


        self.nen4_cv = self.convert_cv_qt(nen4)
        self.uic1.image_calib.setPixmap(self.nen4_cv)
        
    def calib(self):
        print("calibration")
        self.uic1.text_calib.append('Da calib anh')
    def save_calib(self):
        if self.status_cam == True :
            os.makedirs('./image/Calib_image',exist_ok=True)
            cv2.imwrite('./image/Calib_image/'+str(self.calib_dem)+'.jpg',cv_img)
            time.sleep(0.5)
            
            img_calib = cv2.imread('./image/Calib_image/'+str(self.calib_dem)+'.jpg')
            img_calib_cv = self.convert_cv_qt(img_calib)
            self.uic1.image_calib.setPixmap(img_calib_cv)
            self.calib_dem = self.calib_dem + 1
            self.uic1.text_calib.append('Da chup anh '+str(self.calib_dem))
        else:
            self.uic1.text_calib.append('Camera chua ket noi')

    def cam_id_connect(self):
        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False

        if self.cam_id == 0:
            cid = 1
        if self.cam_id == 1:
            cid = 0
        self.cam_id = cid
        self.uic1.text_calib.append('Cam ID : '+str(self.cam_id))
            
    def close_calib(self):
        self.Calib_window.close()

        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False

            nen0 = cv2.imread('./image/display_image/Picture1.png')
            nen0_cv = self.convert_cv_qt(nen0)
            self.uic.cam.setPixmap(nen0_cv)

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    #---------------------------------------------
    def train_show(self):
        self.Train_window = QtWidgets.QMainWindow()
        self.uic3 = Ui_Train_Window()
        self.uic3.setupUi(self.Train_window)
        self.Train_window.show()
        self.uic3.Button_colab.clicked.connect(self.colab)
        self.uic3.Button_save_train.clicked.connect(self.save_train_img)
        self.uic3.Button_close_train.clicked.connect(self.close_train)
        self.uic3.Button_cam_train.clicked.connect(self.make_data)
        self.uic3.Button_reset_train.clicked.connect(self.reset_train)
        self.uic3.Button_labelimg.clicked.connect(self.labelimg)
        self.uic3.Button_temple.clicked.connect(self.select_temple)

        self.uic3.text_train.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        
        self.train_dem = 0
        self.list_img = None

        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False
            
            nen0 = cv2.imread('./image/display_image/Picture1.png')
            nen0_cv = self.convert_cv_qt(nen0)
            self.uic.cam.setPixmap(nen0_cv)

        self.nen4_cv = self.convert_cv_qt(nen4)
        self.uic3.image_train.setPixmap(self.nen4_cv)

    def colab(self):
        url = "https://colab.research.google.com/drive/1eSu2uNAXlFerACXWr8vJNO7io475lko8"
        webbrowser.open_new_tab(url)
        #os.system('python main_auto_train.py')
    def save_train_img(self):
        if self.status_cam == True:
            os.makedirs('./image/Train_image',exist_ok=True)
            cv2.imwrite('./image/Train_image/'+str(self.train_dem)+'.jpg',cv_img)
            time.sleep(0.5)
            
            img_train = cv2.imread('./image/Train_image/'+str(self.train_dem)+'.jpg')
            img_train_cv = self.convert_cv_qt(img_train)
            self.uic3.image_train.setPixmap(img_train_cv)
            self.train_dem = self.train_dem + 1
            self.uic3.text_train.append('Da chup anh '+str(self.train_dem))
        else:
            self.uic3.text_train.append('Camera chua ket noi')

    def reset_train(self):
        self.uic3.text_train.clear()
        self.uic3.text_train.append('Da reset thu muc')
        try:
            shutil.rmtree('./image/train_image')
        except OSError as e:
            print('')

        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False
            self.uic3.image_train.setPixmap(self.nen4_cv)
            
            nen0 = cv2.imread('./image/display_image/Picture1.png')
            nen0_cv = self.convert_cv_qt(nen0)
            self.uic.cam.setPixmap(nen0_cv)
            
        self.train_dem = 0
        self.train_dem1 = 0

    def labelimg(self):
        os.system('labelImg.exe')

    def select_temple(self):
        files = QFileDialog.getOpenFileNames(filter="*.png *.jpg")
        self.list_img = files[0]
        self.tem_image = cv2.imread(self.list_img[0])
        template_image = self.tem_image
        
    def close_train(self):
        self.Train_window.close()

        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False

            nen0 = cv2.imread('./image/display_image/Picture1.png')
            nen0_cv = self.convert_cv_qt(nen0)
            self.uic.cam.setPixmap(nen0_cv)

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    #-----------------------------------------------
    def exit(self):
        self.Exit_window = QtWidgets.QMainWindow()
        self.uic2 = Ui_Exit_window()
        self.uic2.setupUi(self.Exit_window)

        #self.Exit_window.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        
        self.Exit_window.show()
        self.uic2.Button_yes.clicked.connect(self.exit_main_window_1)
        self.uic2.Button_no.clicked.connect(self.exit_exit_window)
        
    #----------------------------------------------------------
    def exit_main_window_1(self):
        if com_status == True:
            Arduino_serial.write('m'.encode())
        #self.stop_capture_video()
        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False

        self.Exit_window.close()
        self.close()

    #---------------------------------------
    def exit_exit_window(self):
        self.Exit_window.close()
        
    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
    #---------------------------------------------------
    def restart_window(self):
        self.Restart_window = QtWidgets.QMainWindow()
        self.uic4 = Ui_Restart_window()
        self.uic4.setupUi(self.Restart_window)

        #self.Restart_window.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        
        self.Restart_window.show()
        self.uic4.Button_yes.clicked.connect(self.exit_main_window_2)
        self.uic4.Button_no.clicked.connect(self.close_restart)

    #--------------------------------------------------------
    def exit_main_window_2(self):
        if com_status == True:
            Arduino_serial.write('m'.encode())
        #self.stop_capture_video()
        if self.status_cam == True:
            self.cap.release()
            self.status_cam = False

        self.Restart_window.close()
        self.close()

    #----------------------------------------------------
    def close_restart(self):
        self.Restart_window.close()

    '''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
        
    #------------------------------------------------   
    def show_image(self):
        #global img11
        img11 = cv2.imread("./image/display_image/anh_yolo.jpg")
        #img11 = cv2.resize(img11, None, fx=0.5, fy=0.5)
        frame11 = self.convert_cv_qt(img11)
        self.uic.image.setPixmap(frame11)
        
        img10 = cv2.imread("./image/display_image/anh_chup.jpg")
        #img10 = cv2.resize(img10, None, fx=0.5, fy=0.5)
        frame10 = self.convert_cv_qt(img10)
        self.uic.image1.setPixmap(frame10)

    #-----------------------------------------------
    def resetScroll(self):
        self.layout = QVBoxLayout()
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.uic.phat_hien_loi.setWidget(self.widget)
    #----------------------------------------------------
    def reset(self):
        if com_status == True:
            Arduino_serial.write('m'.encode())
            nen2 = cv2.imread('./image/display_image/Picture2.png')
            cv2.imwrite('./image/display_image/anh_chup.jpg',nen2)
            cv2.imwrite('./image/display_image/anh_yolo.jpg',nen2)
            self.uic.ket_qua.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 170, 255, 255), stop:1 rgba(255, 255, 164, 255))")
            self.uic.ket_qua.setText('<html><head/><body><p align="center">RESULT </p></body></html>')
            self.show_image()
            self.resetScroll()
            self.uic.message.clear()
            self.uic.message.append('  ----------- Reset ----------')
            self.Board = 0
            
            if self.status_cam == True:
                self.cap.release()
                self.status_cam = False

                nen0 = cv2.imread('./image/display_image/Picture1.png')
                nen0_cv = self.convert_cv_qt(nen0)
                self.uic.cam.setPixmap(nen0_cv)
        else:
            self.restart_window()            
    #-----------------------------------------------------		
    def show_errorImg(self,img,errorName):
        x = img.shape[0]
        y = img.shape[1]
        w = self.uic.phat_hien_loi.width()-50
        
        #Scroll Area Properties
        #self.uic.phat_hien_loi.setWidgetResizable(True)

        temp = QLabel("")
        error_label = QLabel(errorName)
        space = QLabel("   ")
        error_label.setFont(QFont("Arial",18))
        error_label.setAlignment(Qt.AlignTop)
        error_label.setAlignment(Qt.AlignHCenter)
        space.setAlignment(Qt.AlignTop)
        space.setAlignment(Qt.AlignHCenter)
        #error_label.setMinimumSize(QSize(w,20))
        temp.setScaledContents(True)
        qimg = self.convert_cv_qt(img)
        
        temp.setPixmap(qimg)

        temp.setMinimumSize(QSize(y,x))
        temp.setMaximumSize(QSize(y,x))
        temp.setBaseSize(QSize(10,10))

        self.layout.addWidget(temp)
        self.layout.addWidget(error_label)
        #self.layout.addWidget(space)
        
    #---------------------------------------------------
    def show_loi(self):
        self.dem_xuoc = 0
        self.dem_nguoc = 0
        self.dem_thieu = 0
        if i > 0:
            img_xoay = cv2.imread('./image/display_image/anh_xoay.jpg')
            for j in range(i):
                #print(j)
                x1, y1, x2, y2 = boxes[j]               
                img_crop = img_xoay[y1:y2,x1:x2]
                img_crop = cv2.resize(img_crop, None, fx=2, fy=2)
                self.show_errorImg(img_crop,labels[j])
                #print(labels[j])
                if labels[j] == 'xuoc':
                    self.dem_xuoc = self.dem_xuoc + 1
                if labels[j] == 'nguoc':
                    self.dem_nguoc = self.dem_nguoc + 1
                if labels[j] == 'thieu':
                    self.dem_thieu = self.dem_thieu + 1
        self.dem_loi(self.dem_xuoc,self.dem_nguoc,self.dem_thieu)
    #---------------------------------------------------
    def dem_loi(self,dem_xuoc,dem_nguoc,dem_thieu):
        if dem_xuoc :
            self.uic.message.append("  Mạch có "+str(dem_xuoc)+" vết xước ")
        if dem_nguoc :
            self.uic.message.append("  Mạch có "+str(dem_nguoc)+" linh kiện ngược chiều ")
        if dem_thieu :
            self.uic.message.append("  Mạch thiếu "+str(dem_thieu)+" linh kiện ")
    #---------------------------------------------------
    def show_message(self,i):
        if i == 0:
            self.uic.message.append("  Mạch in không có lỗi")
            Arduino_serial.write('t'.encode())
        if i > 0:
            self.uic.message.append("  Mạch in có lỗi")
            Arduino_serial.write('f'.encode())

    #---------------------------------------------------
    def show_ketqua(self,i):
        #self.uic.ket_qua.setStyleSheet("background-color:yellow; border:2px solid black")

        if i == 0:
            #self.uic.ket_qua.setStyleSheet("background-color:green; border:1px solid black")
            self.uic.ket_qua.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 146, 0, 255), stop:1 rgba(85, 255, 0, 255))")
            self.uic.ket_qua.setText('<html><head/><body><p align="center">TRUE </p></body></html>')
        elif i > 0:
            #self.uic.ket_qua.setStyleSheet("background-color:red; border:1px solid black")
            self.uic.ket_qua.setStyleSheet("background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(255, 0, 0, 255), stop:1 rgba(255, 218, 255, 255))")
            self.uic.ket_qua.setText('<html><head/><body><p align="center">FALSE </p></body></html>')

    #--------------------------------------------------------
    def convert_cv_qt(self, cv_img):
        #Convert from an opencv image to QPixmap
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    #------------------------------------------------
    def stop(self):
        if com_status == True:
            Arduino_serial.write('m'.encode())
            self.uic.message.append('  ----------- Stop -----------')
        else:
            self.restart_window()
        
    #------------------------------------------------
    def read_serial(self):
        if com_status == True:
            self.status_cam = True
            self.uic.message.append('  ----------- Start -----------')
            self.cap = cv2.VideoCapture(self.cam_id)
            self.Board = 0
            Arduino_serial.write('n'.encode())
            
            while (self.cap.isOpened()):
                global cv_img
                ret, cv_img = self.cap.read()
                #cv_img = cv2.resize(cv_img, None, fx=1.5, fy=1.5)
                qt_img = self.convert_cv_qt(cv_img)
                self.uic.cam.setPixmap(qt_img)
                cv2.waitKey(2)
                if Arduino_serial.in_waiting:
                    packet = Arduino_serial.read()
                    if(packet.decode('utf').rstrip('\n')=='c'):
                        self.resetScroll()
                        #print("da nhan tin hieu")
                        cv2.imwrite('./image/display_image/anh_chup.jpg',cv_img)
                        img_yl1 = cv2.imread('./image/display_image/anh_chup.jpg')
                        img_yl2 = rotate_image(img_yl1,template_image)
                        cv2.imwrite('./image/display_image/anh_xoay.jpg',img_yl2)
        
                        img_yl3 = yolo(img_yl2)
                        cv2.imwrite('./image/display_image/anh_yolo.jpg',img_yl3)
                        
                        time.sleep(4)
                        
                        self.Board = self.Board + 1
                        self.uic.message.append(' '+str(self.Board)+' --')
                        self.show_message(i)
                        self.show_ketqua(i) 
                        self.show_image()
                        self.show_loi()
        else:
            self.restart_window()

    #----------------------------------------------------
    def make_data(self):
        if com_status == True:
            self.status_cam = True
            self.uic3.text_train.append('Da ket noi camera')
            self.cap = cv2.VideoCapture(self.cam_id)

            Arduino_serial.write('n'.encode())

            self.train_dem1 = 0
            os.makedirs('./image/Train_image',exist_ok=True)
            
            while (self.cap.isOpened()):
                global cv_img
                ret, cv_img = self.cap.read()
                #cv_img = cv2.resize(cv_img, None, fx=1.5, fy=1.5)
                qt_img = self.convert_cv_qt(cv_img)
                self.uic.cam.setPixmap(qt_img)
                cv2.waitKey(2)
                if Arduino_serial.in_waiting:
                    packet = Arduino_serial.read()
                    if(packet.decode('utf').rstrip('\n')=='c'):
                        #print("da nhan tin hieu")
                        cv2.imwrite('./image/Train_image/0_'+str(self.train_dem1)+'.jpg',cv_img)
                        time.sleep(1)
                        self.uic3.text_train.append('Da chup anh 0_'+str(self.train_dem1))

                        img_train_auto = cv2.imread('./image/Train_image/0_'+str(self.train_dem1)+'.jpg')
                        img_train_auto_cv = self.convert_cv_qt(img_train_auto)
                        self.uic3.image_train.setPixmap(img_train_auto_cv)
                        
                        self.train_dem1 = self.train_dem1+1
                Arduino_serial.write('n'.encode())
        else:
            self.restart_window()
    #----------------------------------------------------
    def make_image_calib(self):
        if com_status == True:
            self.status_cam = True
            self.uic1.text_calib.append('Da ket noi camera')
            self.cap = cv2.VideoCapture(self.cam_id)

            Arduino_serial.write('n'.encode())
            
            while (self.cap.isOpened()):
                global cv_img
                ret, cv_img = self.cap.read()
                #cv_img = cv2.resize(cv_img, None, fx=1.5, fy=1.5)
                qt_img = self.convert_cv_qt(cv_img)
                self.uic.cam.setPixmap(qt_img)
                cv2.waitKey(2)
                if Arduino_serial.in_waiting:
                    packet = Arduino_serial.read()
                    if(packet.decode('utf').rstrip('\n')=='c'):
                        #print("da nhan tin hieu")
                        time.sleep(3)
        else:
            self.restart_window()
                    
#--------------------------------------------------------------------------------
def rotate_image(compare_image,template_image):
    #compare_image = cv2.resize(compare_image, None, fx=0.5, fy=0.5)
    #template_image = cv2.resize(template_image, None, fx=0.5, fy=0.5)
    gray_tem = cv2.cvtColor(template_image,cv2.COLOR_BGR2GRAY)
    gray_com = cv2.cvtColor(compare_image,cv2.COLOR_BGR2GRAY)

    orb_fe = cv2.ORB_create()
    kp1,des1 = orb_fe.detectAndCompute(gray_com,None)
    kp2,des2 = orb_fe.detectAndCompute(gray_tem,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    matchedVis = cv2.drawMatches(template_image, kp1, compare_image, kp2,matches[:20], None)
    ptsA = np.zeros((len(matches[:50]), 2), dtype="float")
    ptsB = np.zeros((len(matches[:50]), 2), dtype="float")
    for (i, m) in enumerate(matches[:50]):
        ptsA[i] = kp1[m.queryIdx].pt
        ptsB[i] = kp2[m.trainIdx].pt
    (H, mask) = cv2.findHomography(ptsA,ptsB ,method=cv2.RANSAC)
    (h, w) = template_image.shape[:2]
    aligned = cv2.warpPerspective(compare_image, H, (w, h))

    return aligned
#--------------------------------------------------------------------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = img.shape[:2]  # current shape [height, width]
	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)
#------------------------------------------------------------------------------------
classes_to_filter = None

opt  = {
    
	"weights": "./cfg/best.pt", # Path to weights file default weights are for nano model
	"yaml"   : "./cfg/mydataset.yaml",
	"img-size": 640, # default image size
	"conf-thres": 0.1, # confidence threshold for inference.
	"iou-thres" : 0.45, # NMS IoU threshold for inference.
	"device" : 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
	"classes" : classes_to_filter  # list of classes to filter or None
}
#-----------------------------------------------------------------------------------------
def dec(strr):
    if strr[8] == '.':
        a = int(strr[7:8])
    elif strr[9] == '.':
        a = int(strr[7:9])
    elif strr[10] == '.':
        a = int(strr[7:10])
    elif strr[11] == '.':
        a = int(strr[7:11])
    return a

#-----------------------------------------------------------------------------------------------
def yolo (img0):
    with torch.no_grad():
        global labels
        global boxes
        global i
        
        boxes = []
        labels = []
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
     
        if half:
                model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
     
        if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

	#img0 = cv2.imread(source_image_path)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
	
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
                img = img.unsqueeze(0)
        
	# Inference
        t1 = time_synchronized()
        pred = model(img, augment= False)[0]
	# Apply NMS
        classes = None
        if opt['classes']:
                classes = []
                for class_name in opt['classes']:
                        classes.append(opt['classes'].index(class_name))
        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes= classes, agnostic= False)
        t2 = time_synchronized()

        for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                      
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                        for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        i=0
                        for *xyxy, conf, cls in reversed(det):
                                #label = f'{names[int(cls)]} {conf:.2f}'
                                label = f'{names[int(cls)]}'
                                a = str(xyxy[0])
                                x1 = dec(a)
                                b = str(xyxy[1])
                                y1 = dec(b)
                                c = str(xyxy[2])
                                x2 = dec(c)
                                d = str(xyxy[3])
                                y2 = dec(d)
 
                                labels.append(label)
                                boxes.append([x1, y1, x2, y2])
                                
                                #print(boxes[i])
                                #crop = img0[y1:y2,x1:x2]
                                #cv2.imwrite("Hinh anh xuoc"+str(i),crop)
                                i=i+1
                                #print('al')
                                cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 4)
                                #cv2.imwrite('./image/anhyolo.jpg',img0)
				# plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
    return img0

#---------------------------------------------


#-------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

