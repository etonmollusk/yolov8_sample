from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import cv2
import urllib.request
import numpy as np
import torch


test_img_url = "https://ultralytics.com/images/bus.jpg"
yolo_model = "yolov8n-pose.pt"
trt_enginge = "yolov8n-pose.engine"


def load_test_img_url(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    
    return cv2.imdecode(arr, -1)


def get_model(model_name):
    # Load a model
    model = YOLO(model_name)
    
    # Move model to gpu
    model.to("cuda")

    return model


def convert_model_to_engine():
    # Load a model
    model = YOLO(yolo_model)

    model.export(format="engine")


if __name__ == "__main__":
    img = load_test_img_url(test_img_url)
    model = get_model(yolo_model)

    # Predict with the model
    results = model(img)
    
    annotator = Annotator(img)
    
    # View results
    for r in results:
        for i in range(r.keypoints.shape[0]):
            conf = torch.unsqueeze(r.keypoints.conf[i], dim=1)
            ktps = torch.cat((r.keypoints.xy[i], conf), 1)
            annotator.kpts(ktps, shape=r.keypoints.orig_shape, kpt_line=True)
    
    img = annotator.result()  
    cv2.imshow('yolov8_detection', img)     
    if cv2.waitKey() & 0xFF == 27:
        pass
