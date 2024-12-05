# face_detection.py
import cv2
    

# Load the pre-trained Haar Cascade Classifier
face_detector = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_alt.xml')


def detect_face(image):
    """Detect faces in an image using Haar Cascade Classifier."""
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    faces = face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces
