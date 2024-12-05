# video_capture.py
import cv2
from face_detection import detect_face
from face_recognition import encode_face, recognize_face
from config import VIDEO_SOURCE


def start_video_capture():
    """Start video capture from webcam."""
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None
    return cap


def capture_frame(cap):
    """Capture a frame from the webcam feed."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame


def stop_video_capture(cap):
    """Stop the video capture."""
    cap.release()
    cv2.destroyAllWindows()


def mark_attendance(student_name):
    """Mark attendance for the student (save to file, database, etc.)."""
    print(f"Attendance marked for {student_name}")
