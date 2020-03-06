import cv2
import numpy as np
import math


def merge_picture(img1, img2, dir=0):
    if img1.any() and img2.any():
        shape = img1.shape    #三通道的影像需把-1改成1
        cols = shape[1]
        rows = shape[0]
        channels = shape[2]
        if dir == 0:
            dst = np.zeros((rows * 2 + 2, cols, channels), np.uint8)
            dst[0:rows, 0:cols, :] = img1[0:rows, 0:cols, :]
            dst[rows+2:rows*2+2, 0:cols, :] = img2[0:rows, 0:cols, :]
        if dir == 1:
            dst = np.zeros((rows, cols * 2 + 2, channels), np.uint8)
            dst[0:rows, 0:cols, :] = img1[0:rows, 0:cols, :]
            dst[0:rows, cols+2:cols*2+2, :] = img2[0:rows, 0:cols, :]
        return dst


def Gray2BGR(gray):
    norm_img = np.zeros(gray.shape)
    cv2.normalize(gray , norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    return heat_img


def vector2Euler_angle(rotation_vector, translation_vector, camera_matrix, dist_coeffs, model_points):
    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix,
                                       dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return np.array([float(pitch), float(yaw), float(roll)], dtype="double")

def draw_text_line(img, text_line: str, point, charac_type=cv2.FONT_HERSHEY_COMPLEX, size=0.6, color=(0, 255, 0), thickness=2):

    text_line = text_line.split("\n")
    text_size, baseline = cv2.getTextSize(str(text_line), charac_type, size, thickness)
    for i, text in enumerate(text_line):
        if text:
            # draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            # img = draw_text(img, draw_point, text, drawType)
            cv2.putText(img, text, (point[0], point[1] + (text_size[1] + 2 + baseline) * i), charac_type, size,
                        color, thickness)
    return img

