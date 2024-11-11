

import torch
from ultralytics import YOLO
from roboflow import Roboflow

# YOLOv8 modelini başlat
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli

# Roboflow API anahtarı ile projeyi indir
rf = Roboflow(api_key="UB6G1DerdAEopKHk2UGB")
project = rf.workspace("object-detection-xalah").project("kayisi-siniflandirma-wvvkn")
version = project.version(5)
dataset = version.download("yolov8")


# Modeli eğit
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=100)

