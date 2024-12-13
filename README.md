 Student Attendance System using Face Recognition

 Description

This project implements a student attendance system using face recognition technology. The system uses Cascade Classifier for detecting faces in real-time video streams and FaceNet for recognizing and verifying the identity of students. The MobileNetV2 model is trained using data augmentation techniques to improve accuracy in real-world scenarios.

 Key Features

- Face Detection using Cascade Classifier (Haar Cascade) to detect faces in live video streams.
- Face Recognition to match detected faces with a student database using MobileNetV2.
- Data Augmentation techniques during MobileNetV2 model training, including rotation, brightness adjustment, flipping, etc.
- Real-time Attendance by recognizing faces and marking attendance.

 Installation

1. Clone this repository.
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the FaceNet model:
    ```bash
    python src/train_model.py
    ```

4. Run the system:
    ```bash
    python src/video_capture.py
    ```
5. Setting labrary:
    ```pip install fastapi uvicorn keras tensorflow opencv-python-headless```

6. Run server FastAPI:
    ``` uvicorn api:app --reload```

 Conclusion

This project leverages face recognition technology for efficient and accurate student attendance. With real-time face detection and recognition using Cascade Classifier and MobileNetV2, the system provides an effective solution for modern classrooms.

- Model recognition: ```models/final_model/final_model.h5```