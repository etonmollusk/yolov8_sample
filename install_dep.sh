#!/bin/bash

RED="\033[0;31m"
NC="\033[0m"

ENV_NAME="py310"
YOLO_POSE_MODEL="yolov8n-pose.pt"
YOLO_POSE_MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/${YOLO_POSE_MODEL}"

python_bin=$HOME/.pyenv/versions/${ENV_NAME}/bin/python
yolo_bin=$HOME/.pyenv/versions/${ENV_NAME}/bin/yolo

# Install ultralytics==8.0.176 and onnx
${python_bin} -m pip install --no-deps ultralytics==8.0.176
${python_bin} -m pip install "onnx>=1.12.0"

# Download the pose model
wget ${YOLO_POSE_MODEL_URL}

echo "Converting the example torch model to tensorRT engine..."
${yolo_bin} export model=${YOLO_POSE_MODEL} format=engine
