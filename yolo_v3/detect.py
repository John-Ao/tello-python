#!/usr/bin/env python
import time
from sys import platform
from models import *
import threading
import numpy as np

class Detector():

    def __init__(self,device='cpu'):
        self.model=None
        self.device=torch.device(device)
        self.cfg = '/home/john/Desktop/catkin_ws/src/tello_control/yolo_v3/cfg/yolov3.cfg'
        self.weights = '/home/john/Documents/best-2.pt'
        # self.cfg = 'cfg/yolov3.cfg'
        # self.weights = 'weights/best-2.pt'
        self.img_size=416

    def load_weight(self):
        # Initialize this once

        # device = torch.device('cpu')#torch_utils.select_device(force_cpu=False)
        self.model = Darknet(self.cfg, self.img_size)
        # model.load_state_dict(torch.load(weights, map_location=device)['model'])
        self.model.load_state_dict(torch.load(self.weights, map_location=self.device))
        self.model.to(self.device).eval()
        # return model, device

    def detect_ball(self, img):
        # the input image should be BGR
        # Initialized  for every detection
        img_size=self.img_size
        model=self.model
        # device=self.device
        conf_thres=0.5
        nms_thres=0.5
        # Normalize RGB
        img0 = img

        # Padded resize
        tmpresultimg = self.letterbox(img0, new_shape=img_size)
        img = tmpresultimg[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # move RGB dimention to the front
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        img = torch.from_numpy(img).unsqueeze(0)#.to(device)
        pred, _ = model(img)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
            
        if det is not None and len(det) > 0:
            # Rescale boxes from 416 to true image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            return det.tolist()
        else:
            return
    
    @staticmethod
    def drawbox(img,det):
        img=img.copy()
        # the input image should be BGR
        # Draw bounding boxes and labels of detections
        # Get classes and colors
        classes = ['basketball', 'football', 'volleyball', 'balloon'] # TODO load_classes(parse_data_cfg(data)['names'])
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        for det_pack in det:
            xyxy = [int(i) for i in det_pack[:4]]
            conf = det_pack[4]
            # cls_conf= det_pack[5]
            cls = det_pack[6]

            # Add bbox to the image
            label = '%s %.2f' % (classes[int(cls)], conf)
            plot_one_box(xyxy, img, label=label, color=[255,0,0])
        # cv2.imshow('result',img)
        # cv2.waitKey(3)
        return img

    @staticmethod
    def letterbox(img, new_shape=416, color=(128, 128, 128), mode='auto'):
        # Resize a rectangular image to a 32 pixel multiple rectangle
        # https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]

        if isinstance(new_shape, int):
            ratio = float(new_shape) / max(shape)
        else:
            ratio = max(new_shape) / max(shape)  # ratio  = new / old
        ratiow, ratioh = ratio, ratio
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

        # Compute padding https://github.com/ultralytics/yolov3/issues/232
        if mode is 'auto':  # minimum rectangle
            dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
            dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
        elif mode is 'square':  # square
            dw = (new_shape - new_unpad[0]) / 2  # width padding
            dh = (new_shape - new_unpad[1]) / 2  # height padding
        elif mode is 'rect':  # square
            dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
            dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
        elif mode is 'scaleFill':
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape, new_shape)
            ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)  # INTER_AREA is better, INTER_LINEAR is faster
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return (img, ratiow, ratioh, dw, dh)

    # def detect(self,img):

if __name__ == '__main__':
    img = cv2.imread('data/samples/three_balls.jpg')
    d=Detector()
    d.load_weight()
    # model, device = load_weight()
    start = time.time()
    result_obj=d.detect_ball(img)
    d.drawbox(img,result_obj)
    end = time.time()
    print("time: "+str(end-start)+"s")
    print(result_obj)
    input('Done!')