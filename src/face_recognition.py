# face_recognition.py
import cv2
import numpy as np
from tensorflow import load_model
from config import FACENET_MODEL_PATH

# Load the FaceNet model for face recognition
facenet_model = load_model(FACENET_MODEL_PATH)

def encode_face(face_image):
    """Encode a face into a vector using FaceNet."""
    # Resize face to 160x160 as required by FaceNet
    face_image = cv2.resize(face_image, (160, 160))
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image / 255.0  # Normalize the image
    encoding = facenet_model.predict(face_image)
    return encoding

def recognize_face(face_encoding, known_encodings, known_names):
    """Match the face encoding with known encodings to identify the student."""
    min_dist = 100
    identity = None
    for i, known_encoding in enumerate(known_encodings):
        dist = np.linalg.norm(face_encoding - known_encoding)
        if dist < min_dist:
            min_dist = dist
            identity = known_names[i]
    return identity
