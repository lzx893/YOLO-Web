
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]

# DL model config

        # 'yolov8n': "v8/yolov8.yaml",
        # 'yolov5': "v5/yolov5.yaml",
        # 'yolov6': "v6/yolov6.yaml",
        # 'yolov10': "v10/yolov10.yaml",
        # 'yolov11': "v11/yolov11.yaml",
        # 'Rtdetr': "rt-detr/rtdetr-resnet50.yaml"

DETECTION_MODEL_LIST = [
    "yolov8n",
    "yolo11",
    "yolov5",
    "yolov6",
    "yolov10",
    "Rtdetr"
    
    ]
