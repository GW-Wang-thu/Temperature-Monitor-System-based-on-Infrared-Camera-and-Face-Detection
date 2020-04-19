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
        self.CareAreas = []
        self.Foreheads = []
        self.Temperature = []
        self.AlarmFlags = []
        for i in range(len(self.Anchors)):
            ForeheadAnchor = int(self.Anchors[i][3] + 0.3 * (self.Anchors[i][5] - self.Anchors[i][3]))     # ymin + 0.3 (ymax - ymin)
            self.Foreheads.append(self.Frame[self.Anchors[i][2]+5:self.Anchors[i][4]-5,
                                             self.Anchors[i][3]+5:ForeheadAnchor-5])   # xmin:xmax, ymin:y1
            if self.DistanceCorrectionFlag == 1:
                pass
            if self.GestureCorrectionFlag ==1:
                pass
            self.Temperature.append([np.max(self.Foreheads[i]), # Tem
                                     (np.unravel_index(np.argmax(self.Foreheads[i]), self.Foreheads[i].shape)[0] + self.Anchors[i][2]+5,
                                      np.unravel_index(np.argmax(self.Foreheads[i]), self.Foreheads[i].shape)[1] + self.Anchors[i][3]+5)])  # xmin+idx, ymin+idy
            self.Frame = cv2.rectangle(self.Frame, (self.Anchors[i][2], self.Anchors[i][3]), (self.Anchors[i][4], self.Anchors[i][5]), [255, 0, 0], 2)
            if self.Temperature[i][0] >= self.Alarm_Tem:
                color = [255, 0, 0]
                self.AlarmFlags.append(1)
            else:
                color = [0, 255, 0]
                self.AlarmFlags.append(0)
            self.Frame = cv2.rectangle(self.Frame, (self.Anchors[i][2], self.Anchors[i][3]),
                                       (self.Anchors[i][4], ForeheadAnchor), color, 2)
            self.Frame = cv2.circle(self.Frame, self.Temperature[i][1], 1, color, 4)
            self.Frame = cv2.putText(self.Frame, str(self.Temperature[i][0]),
                                     self.Temperature[i][1],
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

