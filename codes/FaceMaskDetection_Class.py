import cv2
import numpy as np
import torch
import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append('__model/')

class FaceMaskDetector():

    def __init__(self, DCImage, conf_thresh=0.5, iou_thresh = 0.4):
        self.model = torch.load('./__model/face_mask_detection.pth')
        self.anchors = np.expand_dims(np.loadtxt("./__model/anchors_exp.csv", delimiter=","), axis=0)
        self.Frame = DCImage
        self.conf_thresh = conf_thresh,
        self.iou_thresh = iou_thresh,
        self.target_shape = (260, 260)
        self.reshape_Frame()
        self.forward_prop()
        self.decode_bbox()
        self.Get_Result()
        self.Get_Marked_Frame()

    def reshape_Frame(self):
        height, width, _ = self.Frame.shape
        image_resized = cv2.resize(self.Frame, self.target_shape)  # 强行resize到 （260，260，3）
        image_np = image_resized / 255.0  # 归一化到0~1; (260, 260, 3)
        image_exp = np.expand_dims(image_np, axis=0)  # 增加bantch维度; （1，260，260，3）
        self.image_reshaped = image_exp.transpose((0, 3, 1, 2))  # 转置，将通道数变到第二维（batch, channel, height, width)

    def forward_prop(self):
        self.y_bboxes, self.y_scores, = self.model.forward(torch.tensor(self.image_reshaped).float())
        self.y_bboxes = self.y_bboxes.detach().numpy()
        self.y_scores = self.y_scores.detach().numpy()[0]

    def decode_bbox(self):
        variances=[0.1, 0.1, 0.2, 0.2]
        anchor_centers_x = (self.anchors[:, :, 0:1] + self.anchors[:, :, 2:3]) / 2
        anchor_centers_y = (self.anchors[:, :, 1:2] + self.anchors[:, :, 3:]) / 2
        anchors_w = self.anchors[:, :, 2:3] - self.anchors[:, :, 0:1]  # [1, 5972, 4]
        anchors_h = self.anchors[:, :, 3:] - self.anchors[:, :, 1:2]
        raw_outputs_rescale = self.y_bboxes * np.array(variances)
        predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
        predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
        predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
        predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
        predict_xmin = predict_center_x - predict_w / 2
        predict_ymin = predict_center_y - predict_h / 2
        predict_xmax = predict_center_x + predict_w / 2
        predict_ymax = predict_center_y + predict_h / 2
        self.y_bboxes = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)[0]

    def Get_Result(self):
        # To speed up, do single class NMS, not multiple classes NMS.
        self.isFace = np.max(self.y_scores, axis=1)  # 是不是人脸
        self.isMask = np.argmax(self.y_scores, axis=1)  # 带不戴口罩
        keep_top_k=-1
        if len(self.y_bboxes) == 0: return []
        conf_keep_idx = np.where(self.isFace > self.conf_thresh)[0]
        Face_bboxes = self.y_bboxes[conf_keep_idx]
        isFace_confident = self.isFace[conf_keep_idx]
        # Figure Overlap Frames
        pick = []
        xmin = Face_bboxes[:, 0]
        ymin = Face_bboxes[:, 1]
        xmax = Face_bboxes[:, 2]
        ymax = Face_bboxes[:, 3]
        area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
        idxs = np.argsort(isFace_confident)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            # keep top k
            if keep_top_k != -1:
                if len(pick) >= keep_top_k:
                    break
            overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
            overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
            overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
            overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
            overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
            overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
            overlap_area = overlap_w * overlap_h
            overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)
            need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > self.iou_thresh)[0]))
            idxs = np.delete(idxs, need_to_be_deleted_idx)
        # if the number of final bboxes is less than keep_top_k, we need to pad it.
        # TODO
        self.isFace_ID = conf_keep_idx[pick]
        self.isMask_ID = self.isMask[self.isFace_ID]

    def Get_Marked_Frame(self):
        id2class = {0: 'Mask', 1: 'NoMask'}
        self.outputs = []
        for idx in self.isFace_ID:
            conf = float(self.isFace[idx])
            class_id = self.isMask[idx]
            bbox = self.y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * self.Frame.shape[1]))
            ymin = max(0, int(bbox[1] * self.Frame.shape[0]))
            xmax = min(int(bbox[2] * self.Frame.shape[1]), self.Frame.shape[1])
            ymax = min(int(bbox[3] * self.Frame.shape[0]), self.Frame.shape[0])

            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            self.Frame = cv2.rectangle(self.Frame, (xmin, ymin), (xmax, ymax), color, 2)
            self.Frame = cv2.putText(self.Frame, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            self.outputs.append([class_id, conf, xmin, ymin, xmax, ymax])
        self.Frame = cv2.cvtColor(self.Frame, cv2.COLOR_RGB2BGR)

def main():
    imgPath = "./Files/timg.jpg"
    frame = cv2.imread(imgPath)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detect = FaceMaskDetector(frame)
    cv2.imshow("Frame", detect.Frame)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()