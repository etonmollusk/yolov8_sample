from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import cv2
import urllib.request
import numpy as np
import torch
import tensorrt as trt

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.engine.predictor import BasePredictor

from typing import List, Optional, Tuple, Union


test_img_url = "https://ultralytics.com/images/bus.jpg"
yolo_model = "yolov8n-pose.pt"


def load_test_img_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    
    return cv2.imdecode(arr, -1)


def get_torch_model(model_name):
    # Load a model
    model = YOLO(model_name)
    
    # Move model to gpu
    model.to("cuda")

    return model


def get_trt_model(model_name):
    # Load a model
    model = AutoBackend(model_name)#, device=torch.device("cuda"))

    return model


def get_trt_model_shape(model):
    return model.bindings["images"].shape


def convert_model_to_engine():
    # Load a model
    model = YOLO(yolo_model)
    model.export(format="engine")


def create_annotator(img_shape):
    return Annotator(np.zeros(img_shape))


def update_annotator_img(annotator, img):
    annotator.im = img


def annotate_keypoints(annotator, kpt_results):
    for r in kpt_results:
        if r.keypoints.xy.nelement() and r.keypoints.conf != None:
            for i in range(r.keypoints.shape[0]):
                conf = torch.unsqueeze(r.keypoints.conf[i], dim=1)
                ktps = torch.cat((r.keypoints.xy[i], conf), 1)
                annotator.kpts(ktps, shape=r.keypoints.orig_shape, kpt_line=True)


def letterbox(im: np.ndarray, new_shape: Union[Tuple, List] = (640, 640), \
        color: Union[Tuple, List] = (0, 0, 0)) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2] # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
    dw /= 2 # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad: # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=color) # add border

    return im, r, (dw, dh)


def blob(im: np.ndarray, return_seg: bool = False) -> Union[np.ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


if __name__ == "__main__":
    img = load_test_img_url(test_img_url)

    #model = get_torch_model(yolo_model)
    model = get_trt_model("yolov8n-pose.engine")

    shape = get_trt_model_shape(model)
    img, ratio, dwdh = letterbox(img, shape[-2:])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(blob(img)).cuda()
    #dwdh = np.array(dwdh * 2, dtype=np.float32)
    #tensor = np.ascontiguousarray(tensor)

    # Predict
    results = model(tensor)
    #results = model(img)
    
    annotator = create_annotator(img.shape)
    update_annotator_img(annotator, img)

    # View results
    annotate_keypoints(annotator, results)
    img = annotator.result()

    cv2.imshow('yolov8_detection', img)
    if cv2.waitKey() & 0xFF == 27:
        pass
