from ultralytics import YOLO
import cv2
import numpy as np

class FaceDetectorYOLO:
    def __init__(self, model_path="models/yolov8m_200e.pt"):
        """
        Khởi tạo face detector với YOLOv8
        :param model_path: Đường dẫn đến model YOLOv8 đã train cho face detection
        """
        self.model = YOLO(model_path)
        
    def detect_faces(self, image):
        """
        Phát hiện khuôn mặt trong ảnh
        :param image: Ảnh đầu vào (numpy array)
        :return: List các bounding box [(x, y, w, h)]
        """
        results = self.model(image, conf=0.5)[0]  # Confidence threshold 0.5
        boxes = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            boxes.append((x1, y1, w, h))
            
        return boxes

    def draw_faces(self, image, boxes, color=(0, 255, 0), thickness=2):
        """
        Vẽ bounding box lên ảnh
        :param image: Ảnh gốc
        :param boxes: List các bounding box [(x, y, w, h)]
        :return: Ảnh đã vẽ bounding box
        """
        img_draw = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, thickness)
        return img_draw 