from ultralytics import YOLO
model = YOLO('yolov8n-face.pt')
model.download()