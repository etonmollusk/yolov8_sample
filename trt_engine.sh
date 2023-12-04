#!/bin/bash

ENV_NAME="py310"
YOLO_POSE_MODEL="yolov8n-pose.pt"
YOLO_POSE_MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/${YOLO_POSE_MODEL}"

yolo_bin=$HOME/.pyenv/versions/${ENV_NAME}/bin/yolo

# Download the pose model
wget ${YOLO_POSE_MODEL_URL}

echo "Converting the example torch model to tensorRT engine..."
${yolo_bin} export model=${YOLO_POSE_MODEL} format=engine
