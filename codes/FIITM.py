from QTUI.MainWindow import Ui_FIITM
from QTUI.Dialog import Ui_Dialog
from FaceMaskDetection_Class import FaceMaskDetector
from Cameras_Class import Cameras
from TemperatureCalculator_Class import TemCalculator
import utils
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog
from PyQt5.QtCore import QTimer, QCoreApplication, Qt
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import qimage2ndarray
from datetime import datetime
import webbrowser as browser
import os

class FIITM_MAIN(QMainWindow,Ui_FIITM):
    def __init__(self,parent=None):
        super(FIITM_MAIN, self).__init__(parent)
        self.setupUi(self) # son of Ui_Form\setupUi
        # instantiate objects
        self.Cameras = Cameras()
        # function initiations
        self.Timer = QTimer()           # Timer
        self.Parameters_Initiation()    # Initiate Parameters
        self.Widget_Initiation()        # Initiate Widgets
        self.Call_BackgroundFuns()      # Call Background Functions
        self.Timer.timeout.connect(self.Run)  # Call Video Each 1ms 

    def Parameters_Initiation(self):
        # MainWindow
        self.SaveVideoFlag = False
        self.SaveFrameFlag = False
        self.DispDCVideoFlag = False
        self.DispICVideoFlag = False
        self.AutoSaveFramesFlag = False
        self.InputMode = "Cameras"
        self.LastTime = datetime.now()
        # Parameters of Dialog Window
        self.AlarmTem = 37.2
        self.FaceDetectionConf = 0.6
        self.FaceDetectionIOU = 0.3
        self.DistanceCorr_Flag = 0
        self.GestureCorr_Flag = 0
        self.SavePath = "./Files/"
        self.AutosaveFrenquency = 5
        self.CalculateFrenquency = 10
        # Check Status
        self.Log = "Start Application at "+ datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S')+"\nChecking Cameras ...\n"
        self.Loglines = -2
        if self.Cameras.DCCamera.status == False:
            self.Log += "Warning: Wrong with Digital Camera\n"
            self.Loglines += 1
            self.InputMode = "Videos"
        if self.Cameras.IRCamera.status == False:
            self.Log += "Warning: Wrong with Infrared Camera\n"
            self.Loglines += 1
            self.InputMode = "Videos"
        if not os.path.exists("./__model/anchors_exp.csv"):
            self.Log += "Error: Can Not Open ./__model/anchors_exp.csv\n"
            self.Loglines += 1
            self.Save_Log
            QCoreApplication.quit()
        if not (os.path.exists("./__model/face_mask_detection.pth") and os.path.exists("./__model/MainModel.py")):
            self.Log += "Error: Require a Face Detection Model, check ./__model/face_mask_detection.pth and ./__model/MainModel.py\n"
            self.Loglines += 1
            self.Save_Log
            QCoreApplication.quit()
        if not os.path.exists("./Files"):
            os.mkdir("Files")
        # Others
        self.Frame_Num = 0
        self.Video_Num = 0
        self.DCFrame = np.ones((3,480,640),dtype=np.uint8)
        self.ICFrame = np.ones((480,640),dtype=np.uint8)
        self.DC_VideoReader = cv2.VideoCapture(self.Cameras.DCCamera.DC_ID)
        self.IC_VideoReader = cv2.VideoCapture(self.Cameras.IRCamera.IC_ID)
        self.Github_url = "https://baidu.com"
        self.HelpDoc_url = os.path.abspath("./__model/Help.pdf")
        self.Log += "Initiation Done.\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)
        self.ICVideoRVAL = 0
        self.DCVideoRVAL = 0

    def Widget_Initiation(self):
        # Start Timer
        self.Timer.start(1000)
        # Menus and actions(3 + 9)
        self.menuFile.setEnabled(True)
        self.menuSettings.setEnabled(True)
        self.menuHelp.setEnabled(True)
        self.actionOpen_DCPicture.setEnabled(False)
        self.actionOpen_ICPicture.setEnabled(False)
        self.actionOpen_DCVideo.setEnabled(False)
        self.actionOpen_ICVideo.setEnabled(False)
        self.actionSave_Log.setEnabled(True)
        self.actionSave_Settings.setEnabled(True)
        self.actionVisit_Homepage_On_Github.setEnabled(True)
        self.actionHelp_Document.setEnabled(True)
        self.actionCopyright.setEnabled(True)
        # Buttons(9)
        self.RunDC_BT.setEnabled(True)
        self.RunIC_BT.setEnabled(True)
        self.PauseDC_BT.setEnabled(False)
        self.PauseIC_BT.setEnabled(False)
        self.SaveVideoStart_BT.setEnabled(False)
        self.SaveVideoStop_BT.setEnabled(False)
        self.AutosaveFramesStart_BT.setEnabled(False)
        self.AutosaveFramesStop_BT.setEnabled(False)
        self.SaveFrame_BT.setEnabled(False)
        # Radio Buttons(3)
        if (self.Cameras.DCCamera.status and self.Cameras.IRCamera.status):
            self.InputModeCameras_RBT.setChecked(True)
            self.InputModeVideos_RBT.setChecked(False)
            self.InputModePictures_RBT.setChecked(False)
        else:
            self.InputModeCameras_RBT.setChecked(False)
            self.InputModeVideos_RBT.setChecked(True)
            self.InputModeVideos_RBT_Clicked
            self.InputModePictures_RBT.setChecked(False)
            self.InputModeCameras_RBT.setCheckable(False)
            self.actionOpen_DCVideo.setEnabled(True)
            self.actionOpen_ICVideo.setEnabled(True)
        # Texts(3)
        self.DC_Window.setEnabled(True)
        self.IC_Window.setEnabled(True)
        self.LogOut_Window.setReadOnly(True)
        self.LogOut_Window.setOverwriteMode(True)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Call_BackgroundFuns(self):
        # Buttons(9)
        self.RunDC_BT.clicked.connect(self.RunDC_BT_Clicked)
        self.RunIC_BT.clicked.connect(self.RunIC_BT_Clicked)
        self.PauseDC_BT.clicked.connect(self.PauseDC_BT_Clicked)
        self.PauseIC_BT.clicked.connect(self.PauseIC_BT_Clicked)
        self.SaveVideoStart_BT.clicked.connect(self.SaveVideoStart_BT_Clicked)
        self.SaveVideoStop_BT.clicked.connect(self.SaveVideoStop_BT_Clicked)
        self.AutosaveFramesStart_BT.clicked.connect(self.AutosaveFramesStart_BT_Clicked)
        self.AutosaveFramesStop_BT.clicked.connect(self.AutosaveFramesStop_BT_Clicked)
        self.SaveFrame_BT.clicked.connect(self.SaveFrame_BT_Clicked)
        self.Exit_BT.clicked.connect(self.EXIT)
        # Radio Buttons(2)
        self.InputModeCameras_RBT.clicked.connect(self.InputModeCameras_RBT_Clicked)
        self.InputModeVideos_RBT.clicked.connect(self.InputModeVideos_RBT_Clicked)
        self.InputModePictures_RBT.clicked.connect(self.InputModePictures_RBT_Clicked)
        # Menu- QActions(9)
        self.actionSettings.triggered.connect(self.Call_DiaogWindow)
        self.actionHelp_Document_Parameters.triggered.connect(self.Parameters_Explain)
        self.actionOpen_DCPicture.triggered.connect(self.Open_DCPicture)
        self.actionOpen_ICPicture.triggered.connect(self.Open_ICPicture)
        self.actionOpen_DCVideo.triggered.connect(self.Open_DCVideo)
        self.actionOpen_ICVideo.triggered.connect(self.Open_ICVideo)
        self.actionSave_Log.triggered.connect(self.Save_Log)
        self.actionSave_Settings.triggered.connect(self.Save_Settings)
        self.actionVisit_Homepage_On_Github.triggered.connect(self.Visit_Homepage_On_Github)
        self.actionHelp_Document.triggered.connect(self.Help_Document)
        self.actionCopyright.triggered.connect(self.Copyright)

    def EXIT(self):
        QCoreApplication.quit()

    def Call_DiaogWindow(self):
        self.Log += "Parameter Settings\n  {\n"
        self.Loglines += 2
        self.DialogWindow = FIITM_DIALOG()
        print(self.AlarmTem)
        self.DialogWindow.Parameters_Initiation(AlarmTem=self.AlarmTem, Conf=self.FaceDetectionConf,
                                                IOU=self.FaceDetectionIOU, DistanceCorr_Flag=False,
                                                GestureCorr_Flag=self.GestureCorr_Flag, SavePath=self.SavePath,
                                                AutosaveFrenquency=self.AutosaveFrenquency,
                                                CalculateFrenquency=self.CalculateFrenquency)
        self.DialogWindow.exec()
        self.AlarmTem = self.DialogWindow.AlarmTem
        self.FaceDetectionConf = self.DialogWindow.FaceDetectionConf
        self.FaceDetectionIOU = self.DialogWindow.FaceDetectionIOU
        self.DistanceCorr_Flag = self.DialogWindow.DistanceCorr_Flag
        self.GestureCorr_Flag = self.DialogWindow.GestureCorr_Flag
        self.SavePath = self.DialogWindow.SavePath
        self.AutosaveFrenquency = self.DialogWindow.AutosaveFrenquency
        self.CalculateFrenquency = self.DialogWindow.CalculateFrenquency
        self.Log += self.DialogWindow.Log + "\n  }"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def RunDC_BT_Clicked(self):
        # Related Changes
        self.DispDCVideoFlag = True
        self.RunDC_BT.setEnabled(False)
        self.PauseDC_BT.setEnabled(True)
        self.SaveFrame_BT.setEnabled(True)
        if self.DispICVideoFlag and not self.SaveVideoFlag:
            self.SaveVideoStart_BT.setEnabled(True)
            self.AutosaveFramesStart_BT.setEnabled(True)
            self.SaveVideoStop_BT.setEnabled(False)
            self.AutosaveFramesStop_BT.setEnabled(False)
        self.SaveFrame_BT.setEnabled(True)
        # Actions
        self.Log += "Run Digital Camera\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)
    
    def RunIC_BT_Clicked(self):
        # Related Changes
        self.DispICVideoFlag = True
        self.RunIC_BT.setEnabled(False)
        self.PauseIC_BT.setEnabled(True)
        self.SaveFrame_BT.setEnabled(True)
        if self.DispDCVideoFlag and not self.SaveVideoFlag:
            self.SaveVideoStart_BT.setEnabled(True)
            self.AutosaveFramesStart_BT.setEnabled(True)
            self.SaveVideoStop_BT.setEnabled(False)
            self.AutosaveFramesStop_BT.setEnabled(False)
        self.SaveFrame_BT.setEnabled(True)
        # Actions
        self.Log += "Run Infrared Camera\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def PauseDC_BT_Clicked(self):
        # Related Changes
        self.DispDCVideoFlag = False
        self.SaveVideoStart_BT.setEnabled(False)
        self.SaveVideoStop_BT.setEnabled(False)
        self.PauseDC_BT.setEnabled(False)
        if self.DCFrame.any():
            self.RunDC_BT.setEnabled(True)
        if not self.DispICVideoFlag and not self.SaveVideoFlag:
            self.InputModeCameras_RBT.setEnabled(True)
            self.InputModeVideos_RBT.setEnabled(True)
            self.InputModePictures_RBT.setEnabled(True)
        # Actions
        self.Log += "Stop Digital Camera\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def PauseIC_BT_Clicked(self):
        # Related Changes
        self.DispICVideoFlag = False
        self.SaveVideoStart_BT.setEnabled(False)
        self.SaveVideoStop_BT.setEnabled(False)
        self.PauseIC_BT.setEnabled(False)
        if self.ICFrame.any():
            self.RunIC_BT.setEnabled(True)
        if not self.DispDCVideoFlag and not self.SaveVideoFlag:
            self.InputModeCameras_RBT.setEnabled(True)
            self.InputModeVideos_RBT.setEnabled(True)
            self.InputModePictures_RBT.setEnabled(True)
        # Actions
        self.Log += "Stop Infrared Camera\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def SaveVideoStart_BT_Clicked(self):
        # Related Changes
        if self.MergeFrame.any():
            self.SaveVideoFlag = True
            self.SaveVideoStart_BT.setEnabled(False)
            self.SaveVideoStop_BT.setEnabled(True)
            self.PauseDC_BT.setEnabled(False)
            self.PauseIC_BT.setEnabled(False)
            # Actions
            self.Video_Num += 1
            CurrentTime = datetime.now()
            self.Video_Name = self.SavePath + np.str(self.Video_Num) +"_"+ np.str(CurrentTime.hour)+"_"+np.str(CurrentTime.minute)+"_" +np.str(CurrentTime.second)+".avi"
            self.VideoWriter = cv2.VideoWriter()
            size = self.MergeFrame.shape
            self.VideoWriter.open(self.Video_Name, cv2.VideoWriter_fourcc('M','J','P','G'),int(self.CalculateFrenquency), (int(size[1]), int(size[0])), True)
            self.Log += "Strat Saving Video "+ np.str(self.Video_Num) +" at" + datetime.strftime(CurrentTime,'%Y-%m-%d %H:%M:%S') + "\n"
            self.Loglines += 1
            self.LogOut_Window.setPlainText(self.Log)
            self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def AutosaveFramesStart_BT_Clicked(self):
        # Related Changes
        self.AutoSaveFramesFlag = True
        self.AutosaveFramesStart_BT.setEnabled(False)
        self.AutosaveFramesStop_BT.setEnabled(True)
        self.PauseDC_BT.setEnabled(False)
        self.PauseIC_BT.setEnabled(False)
        # Actions
        self.CurrentTime = datetime.now()
        self.Log += "Strat Autosave Frames "+ str(self.AutosaveFrenquency) +" f/min at" + datetime.strftime(self.CurrentTime,'%Y-%m-%d %H:%M:%S') + "\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def SaveVideoStop_BT_Clicked(self):
        # Related Changes
        self.SaveVideoFlag = False
        self.SaveVideoStart_BT.setEnabled(True)
        self.SaveVideoStop_BT.setEnabled(False)
        if not self.AutoSaveFramesFlag:
            self.PauseDC_BT.setEnabled(True)
            self.PauseIC_BT.setEnabled(True)
        # Actions
        self.VideoWriter.release()
        self.Log += "Stop Saving Video at " + datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S') + "\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def AutosaveFramesStop_BT_Clicked (self):
        # Related Changes
        self.AutoSaveFramesFlag = False
        self.AutosaveFramesStart_BT.setEnabled(True)
        self.AutosaveFramesStop_BT.setEnabled(False)
        if not self.SaveVideoFlag:
            self.PauseDC_BT.setEnabled(True)
            self.PauseIC_BT.setEnabled(True)
        # Actions
        self.Log += "Stop AutoSaving Frames at " + datetime.strftime(datetime.now(),'%Y-%m-%d %H:%M:%S') + "\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def SaveFrame_BT_Clicked(self):
        # Actions
        self.Frame_Num += 1
        CurrentTime = datetime.now()
        self.CurrentTime = CurrentTime
        self.Frame_Name = self.SavePath + np.str(self.Frame_Num) +"_"+ np.str(CurrentTime.hour)+"_"+np.str(CurrentTime.minute)+"_" +np.str(CurrentTime.second)
        DCimg = cv2.cvtColor(self.DCFrame, cv2.COLOR_BGR2RGB)
        ICimg = cv2.cvtColor(self.ICFrame, cv2.COLOR_BGR2RGB)
        DCimg = cv2.resize(DCimg, (640, 480))
        ICimg = cv2.resize(ICimg, (640, 480))
        cv2.imwrite(self.Frame_Name + "_DC.jpg", DCimg)
        cv2.imwrite(self.Frame_Name + "_IC.jpg", ICimg)
        self.Log += "Save Frames to "+ self.Frame_Name +" at " + datetime.strftime(CurrentTime,'%Y-%m-%d %H:%M:%S') + "\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def InputModeCameras_RBT_Clicked(self):
        # Related Changes
        self.actionOpen_DCPicture.setEnabled(False)
        self.actionOpen_ICPicture.setEnabled(False)
        self.actionOpen_DCVideo.setEnabled(False)
        self.actionOpen_ICVideo.setEnabled(False)
        self.InputMode = "Cameras"
        self.PauseDC_BT_Clicked()
        self.PauseIC_BT_Clicked()
        self.DCFrame = np.ones((3,480,640),dtype=np.uint8)
        self.ICFrame = np.ones((3,480,640),dtype=np.uint8)
        self.DC_VideoReader = cv2.VideoCapture(self.Cameras.DCCamera.DC_ID)
        self.IC_VideoReader = cv2.VideoCapture(self.Cameras.IRCamera.IC_ID)
        self.InputModePictures_RBT.setChecked(False)
        self.InputModeVideos_RBT.setChecked(False)
        # Actions
        self.Log += "Set Input: From Cameras\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def InputModeVideos_RBT_Clicked(self):
        # Related Changes
        self.DC_VideoReader.release()
        self.IC_VideoReader.release()
        self.actionOpen_DCPicture.setEnabled(False)
        self.actionOpen_ICPicture.setEnabled(False)
        self.actionOpen_DCVideo.setEnabled(True)
        self.actionOpen_ICVideo.setEnabled(True)
        self.InputMode = "Videos"
        self.DCFrame = np.ones((3,480,640),dtype=np.uint8)
        self.ICFrame = np.ones((3,480,640),dtype=np.uint8)
        self.PauseDC_BT_Clicked()
        self.PauseIC_BT_Clicked()
        self.RunDC_BT.setEnabled(False)
        self.RunIC_BT.setEnabled(False)
        self.InputModeCameras_RBT.setChecked(False)
        self.InputModePictures_RBT.setChecked(False)
        # Actions
        self.Log += "Set Input: From Videos\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)
        
    def InputModePictures_RBT_Clicked(self):
        # Related Changes
        self.DC_VideoReader.release()
        self.IC_VideoReader.release()
        self.actionOpen_DCPicture.setEnabled(True)
        self.actionOpen_ICPicture.setEnabled(True)
        self.actionOpen_DCVideo.setEnabled(False)
        self.actionOpen_ICVideo.setEnabled(False)
        self.InputMode = "Pictures"
        self.RunDC_BT.setEnabled(False)
        self.RunIC_BT.setEnabled(False)
        self.PauseDC_BT.setEnabled(False)
        self.PauseIC_BT.setEnabled(False)
        self.SaveVideoStart_BT.setEnabled(False)
        self.SaveVideoStop_BT.setEnabled(False)
        self.AutosaveFramesStart_BT.setEnabled(False)
        self.AutosaveFramesStop_BT.setEnabled(False)
        self.DispICVideoFlag = False
        self.DispDCVideoFlag = False
        self.InputModeCameras_RBT.setChecked(False)
        self.InputModeVideos_RBT.setChecked(False)
        self.DCFrame = np.ones((3,480,640),dtype=np.uint8)
        self.ICFrame = np.ones((3,480,640),dtype=np.uint8)
        self.SaveFrame_BT.setEnabled(True)
        # Actions
        self.Log += "Set Input: From Picturess\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Open_DCPicture(self):
        # Actions
        fileName, filetype = QFileDialog.getOpenFileName(self, "Open DCPictures","./","All Files (*);;Text Files (*.jpg, *.bmp, *.png, *.tif)")
        if fileName:
            self.DCFrame = cv2.imread(fileName)
            self.DCFrame = cv2.resize(self.DCFrame, (640, 480))
            self.FaceDetector = FaceMaskDetector(self.DCFrame,
                                                 conf_thresh=self.FaceDetectionConf,
                                                 iou_thresh=self.FaceDetectionIOU)
            self.DCFrame = self.FaceDetector.Frame
            print(self.DCFrame[240, 320, 1])
            self.DispFrame("DC")
            self.Log += "Open Digital Picture: " + fileName + "\n"
            self.Loglines += 1
            self.LogOut_Window.setPlainText(self.Log)
            self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Open_ICPicture(self):
        # Actions
        fileName, filetype = QFileDialog.getOpenFileName(self, "Open ICPictures","./","CSV Files (*.csv);;Text Files (*.jpg, *.bmp, *.png, *.tif)")
        if fileName:
            self.ICFrame = np.loadtxt(fileName, delimiter=",")
            self.ICFrame = utils.Gray2BGR(self.ICFrame)
            self.ICFrame = cv2.resize(self.ICFrame, (640, 480), interpolation=cv2.INTER_CUBIC)
            print(self.ICFrame[240, 320, 1])
            if self.ICFrame[240, 320, 1]:
                self.DCVideoRVAL = True
                self.TempCalculator = TemCalculator(self.ICFrame,
                                                    Anchors=self.FaceDetector.outputs,
                                                    Alarm_Tem=self.AlarmTem,
                                                    DistanceCorrectionFlag=self.DistanceCorr_Flag,
                                                    GestureCorrectionFlag=self.GestureCorr_Flag)
                self.ICFrame = self.TempCalculator.LabeledImage
                self.DispFrame("IC")
                self.Log += "Open Infrared Picture: " + fileName + "\n"
                self.Loglines += 1
            else:
                self.Log += "Open Corresponding Digital Image First ! \n"
                self.Loglines += 1
            self.LogOut_Window.setPlainText(self.Log)
            self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Open_DCVideo(self):
        # Related Changes
        fileName, filetype = QFileDialog.getOpenFileName(self, "Open DCVideos","./","Video Files (*.avi)")
        if fileName:
            self.ICVideoRVAL = min(self.ICVideoRVAL, 1)
            self.DC_VideoReader = cv2.VideoCapture(fileName)
            self.DCVideoRVAL_Max = self.DC_VideoReader.get(cv2.CAP_PROP_FRAME_COUNT)
            self.DC_VideoReader.set(cv2.CAP_PROP_POS_FRAMES, 1)
            self.DCVideoRVAL, self.DCFrame = self.DC_VideoReader.read()
            print(self.DCVideoRVAL)
            self.RunDC_BT.setEnabled(True)
            # Actions
            self.Log += "Open Digital Video: " + fileName + " successfully\n"
            self.Loglines += 1
            self.LogOut_Window.setPlainText(self.Log)
            self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Open_ICVideo(self):
        # Related Changes
        fileName, filetype = QFileDialog.getOpenFileName(self, "Open DCVideos","./","Video Files (*.avi)")
        if fileName:
            self.DCVideoRVAL = min(self.DCVideoRVAL, 1)
            self.IC_VideoReader = cv2.VideoCapture(fileName)
            self.ICVideoRVAL_Max = self.IC_VideoReader.get(cv2.CAP_PROP_FRAME_COUNT)
            self.IC_VideoReader.set(cv2.CAP_PROP_POS_FRAMES, 1)
            self.ICVideoRVAL, self.ICFrame = self.IC_VideoReader.read()
            print(self.ICVideoRVAL)
            self.RunIC_BT.setEnabled(True)
            # Actions
            self.Log += "Open Infrared Video: " + fileName + " successfully\n"
            self.Loglines += 1
            self.LogOut_Window.setPlainText(self.Log)
            self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Save_Log(self):
        # Actions
        fh = open(".\Files\Log.log", 'w')
        fh.write(self.Log)
        fh.close()
        self.Log += "Save Log To .\Files\Log.log\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Save_Settings(self):
        # Actions
        fh = open(".\Files\Settings.sets", 'w')
        fh.write("Alarm Temperature: " + str(self.AlarmTem))
        fh.write("\nFace Detection Conf Tolarance: " + str(self.FaceDetectionConf))
        fh.write("\nFace Detection IOU Tolarance: " + str(self.AlarmTem))
        fh.write("\nEnable Distance Correction: " + str(self.DistanceCorr_Flag))
        fh.write("\nEnable Distance Correction: " + str(self.GestureCorr_Flag))
        fh.write("\nFrequency of Autosave Frames: " + str(self.AutosaveFrenquency))
        fh.write("\nVideo Calculate Frequency: " + str(self.CalculateFrenquency))
        fh.close()
        self.Log += "Save Settings To .\Files\Settings.sets\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Visit_Homepage_On_Github(self):
        browser.open(self.Github_url)

    def Help_Document(self):
        browser.open(self.HelpDoc_url)

    def Copyright(self):
        self.Log += "Copyright@GuowenWang, Tsinghua University\n"
        self.Loglines += 1
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def Parameters_Explain(self):
        self.Log += "View Help Doc"
        self.Loglines += 1
        browser.open(self.HelpDoc_url)
        self.LogOut_Window.setPlainText(self.Log)
        self.LogOut_Window.verticalScrollBar().setSliderPosition(self.Loglines)

    def DispFrame(self, cmd):
        if cmd == "DC" and (self.DCFrame.any()):
            self.DCFrame = cv2.resize(self.DCFrame, (640, 480))
            Qframe = qimage2ndarray.array2qimage(self.DCFrame)
            self.DC_Window.setPixmap(QPixmap(Qframe))
            self.DC_Window.setScaledContents(True)
            self.DC_Window.show()
        elif cmd == "IC" and (self.ICFrame.any()):
            self.ICFrame = cv2.resize(self.ICFrame, (640, 480))
            Qframe = qimage2ndarray.array2qimage(self.ICFrame)
            self.IC_Window.setPixmap(QPixmap(Qframe))
            self.IC_Window.setScaledContents(True)
            self.IC_Window.show()

    def Run(self):
        if self.InputMode == "Cameras":
            _, self.DCFrame = self.DC_VideoReader.read()
            _, self.ICFrame = self.IC_VideoReader.read()
            self.FaceDetector = FaceMaskDetector(self.DCFrame, 
                                                 conf_thresh=self.FaceDetectionConf,
                                                 iou_thresh=self.FaceDetectionIOU)
            self.DCFrame = self.FaceDetector.Frame
            self.TempCalculator = TemCalculator(self.ICFrame, 
                                                Anchors=self.FaceDetector.outputs, 
                                                Alarm_Tem=self.AlarmTem, 
                                                DistanceCorrectionFlag=self.DistanceCorr_Flag, 
                                                GestureCorrectionFlag=self.GestureCorr_Flag)
            self.ICFrame = self.TempCalculator.LabeledImage
            if self.DispDCVideoFlag:
                self.DispFrame("DC")
            if self.DispICVideoFlag:
                self.DispFrame("IC")
                if self.DispDCVideoFlag:
                    self.MergeFrame = utils.merge_picture(self.DCFrame, self.ICFrame, dir=1)
            
        if self.InputMode == "Videos":
            if self.DispDCVideoFlag and self.DCVideoRVAL and (self.DCVideoRVAL < self.DCVideoRVAL_Max-1):
                self.DCVideoRVAL += 1
                self.DC_VideoReader.set(cv2.CAP_PROP_POS_FRAMES, max(self.ICVideoRVAL,self.DCVideoRVAL))
                DC_success, self.DCFrame = self.DC_VideoReader.read()
                if DC_success:
                    self.FaceDetector = FaceMaskDetector(self.DCFrame,
                                                         conf_thresh=self.FaceDetectionConf,
                                                         iou_thresh=self.FaceDetectionIOU)
                    self.DCFrame = self.FaceDetector.Frame
                    self.DispFrame("DC")
            if self.DispICVideoFlag and self.ICVideoRVAL and (self.ICVideoRVAL < self.ICVideoRVAL_Max-1):
                self.ICVideoRVAL += 1
                self.IC_VideoReader.set(cv2.CAP_PROP_POS_FRAMES, max(self.ICVideoRVAL,self.DCVideoRVAL))
                IC_success, self.ICFrame = self.IC_VideoReader.read()
                if IC_success:
                    self.TempCalculator = TemCalculator(self.ICFrame,
                                                        Anchors=self.FaceDetector.outputs,
                                                        Alarm_Tem=self.AlarmTem,
                                                        DistanceCorrectionFlag=self.DistanceCorr_Flag,
                                                        GestureCorrectionFlag=self.GestureCorr_Flag)
                    self.ICFrame = self.TempCalculator.LabeledImage
                    self.DispFrame("IC")
                    if self.DispDCVideoFlag:
                        self.MergeFrame = utils.merge_picture(self.DCFrame, self.ICFrame, dir=1)
        if self.SaveVideoFlag:
            self.VideoWriter.write(self.MergeFrame)
        if self.AutoSaveFramesFlag:
            now = datetime.now()
            if (now.minute * 60 + now.second) - (self.CurrentTime.minute * 60 + self.CurrentTime.second) >= (60 / self.AutosaveFrenquency):
                self.SaveFrame_BT_Clicked()
        self.Timer.start(1000 / self.CalculateFrenquency)

class FIITM_DIALOG(QDialog, Ui_Dialog):

    def __init__(self, parent=None):
        super(FIITM_DIALOG, self).__init__(parent)
        self.setupUi(self) # son of Ui_Form\setupUi
        # function initiations
        self.Parameters_Initiation()
        self.Widget_Initiation()        # Initiate Widgets
        self.Call_BackgroundFuns()      # Call Background Functions
        
    def Parameters_Initiation(self, AlarmTem=37.5, Conf=0.5, IOU=0.3, DistanceCorr_Flag=0, GestureCorr_Flag=0, SavePath="./File/", AutosaveFrenquency=10, CalculateFrenquency=10):
        self.AlarmTem = AlarmTem
        self.FaceDetectionConf = Conf
        self.FaceDetectionIOU = IOU
        self.DistanceCorr_Flag = DistanceCorr_Flag
        self.GestureCorr_Flag = GestureCorr_Flag
        self.SavePath = SavePath
        self.AutosaveFrenquency = AutosaveFrenquency
        self.CalculateFrenquency = CalculateFrenquency
        self.Log = ""
        self.AlarmTem_Text.setText(str(AlarmTem))
        self.Conf_Text.setText(str(Conf))
        self.IOU_Text.setText(str(IOU))
        self.DistanceCorrection_RBT.setChecked(DistanceCorr_Flag)
        self.FacialPoseCorrection_RBT.setChecked(GestureCorr_Flag)
        self.FilePath_Text.setText(SavePath)
        self.AutosaveFrameFrequency_Text.setText(str(AutosaveFrenquency))
        self.VideoCalculateFrequency_Text.setText(str(CalculateFrenquency))
        self.Parameters_Copy(key=0)

    def Parameters_Copy(self, key):
        if key == 0:
            self.__AlarmTem = self.AlarmTem
            self.__FaceDetectionConf = self.FaceDetectionConf
            self.__FaceDetectionIOU =  self.FaceDetectionIOU
            self.__DistanceCorr_Flag = self.DistanceCorr_Flag
            self.__GestureCorr_Flag = self.GestureCorr_Flag
            self.__SavePath = self.SavePath
            self.__AutosaveFrenquency = self.AutosaveFrenquency
            self.__CalculateFrenquency = self.CalculateFrenquency
        if key == 1:
            self.AlarmTem = self.__AlarmTem
            self.FaceDetectionConf = self.__FaceDetectionConf
            self.FaceDetectionIOU = self.__FaceDetectionIOU
            self.DistanceCorr_Flag = self.__DistanceCorr_Flag
            self.GestureCorr_Flag = self.__GestureCorr_Flag
            self.SavePath = self.__SavePath
            self.AutosaveFrenquency = self.__AutosaveFrenquency
            self.CalculateFrenquency = self.__CalculateFrenquency
            self.__Log = ""

    def Widget_Initiation(self):
        # Enable and Show Current Parameters
        # Buttons(3)
        self.DistanceCorrection_RBT.setEnabled(True)
        self.DistanceCorrection_RBT.setChecked(self.DistanceCorr_Flag)
        self.FacialPoseCorrection_RBT.setEnabled(True)
        self.FacialPoseCorrection_RBT.setChecked(self.GestureCorr_Flag)
        self.Browser_BT.setEnabled(True)
        # Texts(3)
        self.AlarmTem_Text.setEnabled(True)
        self.AlarmTem_Text.setText(str(self.AlarmTem))
        self.Conf_Text.setEnabled(True)
        self.Conf_Text.setText(str(self.FaceDetectionConf))
        self.IOU_Text.setEnabled(True)
        self.IOU_Text.setText(str(self.FaceDetectionIOU))
        self.FilePath_Text.setEnabled(True)
        self.FilePath_Text.setText(self.SavePath)
        self.AutosaveFrameFrequency_Text.setEnabled(True)
        self.AutosaveFrameFrequency_Text.setText(str(self.AutosaveFrenquency))
        self.VideoCalculateFrequency_Text.setEnabled(True)
        self.VideoCalculateFrequency_Text.setText(str(self.CalculateFrenquency))

    def Call_BackgroundFuns(self):
        # Buttons
        self.DistanceCorrection_RBT.clicked.connect(self.EnableDistanceCorrection)
        self.FacialPoseCorrection_RBT.clicked.connect(self.EnableGestureCorrection)
        self.Browser_BT.clicked.connect(self.Browser)
        # Texts
        self.AlarmTem_Text.textChanged.connect(self.SetAlarmTem)
        self.Conf_Text.textChanged.connect(self.SetConf)
        self.IOU_Text.textChanged.connect(self.SetIOU)
        self.AutosaveFrameFrequency_Text.textChanged.connect(self.SetAutosaveFrenqucy)
        self.VideoCalculateFrequency_Text.textChanged.connect(self.SetCalculateFrenqucy)

    def EnableDistanceCorrection(self):
        self.DistanceCorr_Flag = True
        self.Log += "Enable Distance Correction\n"
        self.Loglines += 1

    def EnableGestureCorrection(self):
        self.GestureCorr_Flag = True
        self.Log += "Enable Facial Gesture Correction\n"
        self.Loglines += 1

    def Browser(self):
        dirname = QFileDialog.getExistingDirectory(self, "Save to...", '.')
        self.SavePath = dirname + '/'
        self.FilePath_Text.setText(self.SavePath)
        self.Log += "Change Save Directory to " + dirname + '\n'
        self.Loglines += 1

    def SetAlarmTem(self):
        self.AlarmTem = np.float(self.AlarmTem_Text.text())
        self.Log += "Change Alarm Temperature to " + str(self.AlarmTem_Text) +"℃\n"
        self.Loglines += 1

    def SetConf(self):
        self.FaceDetectionConf = np.float(self.Conf_Text.text())
        self.Log += "Change Face Detection Confidence Tolarance to " + str(self.FaceDetectionConf) + "\n"
        self.Loglines += 1

    def SetIOU(self):
        self.FaceDetectionIOU = np.float(self.IOU_Text.text())
        self.Log += "Change Face Detection IOU Tolarance to " + str(self.FaceDetectionIOU) + "\n"
        self.Loglines += 1

    def SetAutosaveFrenqucy(self):
        self.AutosaveFrenquency = np.int(self.AutosaveFrameFrequency_Text.text())
        self.Log += "Set Autosave Frequency at " + str(self.AutosaveFrenquency) + "f/min\n"
        self.Loglines += 1

    def SetCalculateFrenqucy(self):
        self.CalculateFrenquency = np.int(self.VideoCalculateFrequency_Text.text())
        self.Log += "Set Calculate Frequency at " + str(self.CalculateFrenquency) + "times/s\n"
        self.Loglines += 1

    def accept(self):
        self.Log += ""
        print("accept")
        self.done(1)

    def reject(self):
        self.Parameters_Copy(key=1)
        self.Log += "Cancel Settings"
        self.Loglines += 1
        self.done(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MAIN_UI = FIITM_MAIN()
    MAIN_UI.show()
    sys.exit(app.exec_())