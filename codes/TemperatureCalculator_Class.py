import numpy as np
import cv2
import utils

class TemCalculator():
    def __init__(self, IRImage, Anchors, Alarm_Tem = 37, DistanceCorrectionFlag = 0, GestureCorrectionFlag = 0):
        self.Frame = IRImage
        self.LabeledImage = IRImage
        self.Anchors = Anchors
        self.Alarm_Tem = Alarm_Tem
        self.DistanceCorrectionFlag = 0
        self.GestureCorrectionFlag = 0
        self.CalculateTem()
    
    def CalculateTem(self):
        color = (0, 255, 0)
        for i in range(len(self.Anchors)):
            self.LabeledImage = cv2.rectangle(self.Frame, (self.Anchors[i][2], self.Anchors[i][3]), (self.Anchors[i][4], self.Anchors[i][5]), color, 2)
