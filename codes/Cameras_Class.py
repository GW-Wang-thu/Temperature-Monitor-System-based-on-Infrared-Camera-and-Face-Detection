import cv2
import dlib
import numpy as np
import utils

class DCCameras():
    def __init__(self):
        self.DC_ID = 0
        self.status = True
        self.focal_length = 640
        self.resolution = (480, 640)
        self.center = (240, 320)
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        self.camera_matrix = np.array([ [self.focal_length, 0, self.center[0]],
                                        [0, self.focal_length, self.center[1]],
                                        [0, 0, 1]], dtype="double")


class IRCameras():
    def __init__(self):
        self.IC_ID = "http://admin:admin@192.168.1.100:8081/"
        self.status = True
        # self.temperature_range = []   # 温度范围
        # self.epsilon = 0.8    # 发射率
        # self.position_factor = 0.01   # 人脸朝向系数
        # self.distance_factor = 0.9    # 距离影响因子
        self.resolution = (480, 640)
        self.focal_length = 640
        self.center = (240, 320)
        self.camera_matrix = np.array([[self.focal_length, 0, self.center[0]],
                                       [0, self.focal_length, self.center[1]],
                                       [0, 0, 1]], dtype="double")


class Cameras():
    def __init__(self):
        self.DCCamera = DCCameras()
        self.IRCamera = IRCameras()
