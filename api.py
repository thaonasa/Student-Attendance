from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import numpy as np
import cv2
import os
from typing import List

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Định cấu hình CORS nếu cần thiết
app.add_middleware(
    CORSMiddleware,
    # Cho phép tất cả các nguồn, hoặc thay bằng danh sách nguồn cụ thể
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn model đã train
MODEL_PATH = "models/final_model/final_model.h5"
model = load_model(MODEL_PATH)  # Tải model
# Thay bằng tên các lớp của bạn
class_labels = ["Label_1", "Label_2", "Label_3"]


@app.get("/")
def root():
    return {"message": "Welcome to the Face Recognition API!"}


@app.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    """
    Nhận ảnh khuôn mặt từ client, chạy nhận diện và trả về kết quả.
    """
    try:
        # Đọc ảnh từ file upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400, detail="Invalid image format.")

        # Tiền xử lý ảnh
        # Kích thước input của model
        image_resized = cv2.resize(image, (160, 160))
        image_normalized = image_resized / 255.0  # Chuẩn hóa
        image_expanded = np.expand_dims(
            image_normalized, axis=0)  # Thêm batch dimension

        # Dự đoán
        predictions = model.predict(image_expanded)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Kết quả nhận diện
        result = {
            "predicted_class": class_labels[predicted_class],
            "confidence": float(confidence),
        }
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    """
    Nhận ảnh từ client, phát hiện khuôn mặt và trả về các bounding box.
    """
    try:
        # Đọc ảnh từ file upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400, detail="Invalid image format.")

        # Sử dụng Haarcascade để phát hiện khuôn mặt
        cascade_path = "haarcascades/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Trả về kết quả bounding box
        results = []
        for (x, y, w, h) in faces:
            results.append({"x": int(x), "y": int(
                y), "width": int(w), "height": int(h)})

        if not results:
            return {"message": "No faces detected."}
        return {"faces": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
