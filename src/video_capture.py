from config import VIDEO_SOURCE
from face_detection import detect_faces, load_haar_cascade
from face_recognition import recognize_face, save_unknown_face, add_new_student
import cv2
import sys
import os

# Thêm đường dẫn gốc dự án vào sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def display_faces(frame, faces):
    """
    Phát hiện và nhận diện khuôn mặt, sau đó hiển thị thông tin lên khung hình.
    :param frame: Khung hình từ video.
    :param faces: Danh sách tọa độ khuôn mặt.
    """
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y+h, x:x+w]  # Cắt khuôn mặt từ frame

        # Gọi hàm nhận diện khuôn mặt
        student_id, confidence = recognize_face(cropped_face)

        # Xử lý khuôn mặt chưa nhận diện
        if not student_id:
            print("Unknown face detected.")
            image_path = save_unknown_face(
                cropped_face)  # Lưu ảnh chưa nhận diện
            add_new_student(image_path)  # Yêu cầu nhập thông tin và lưu vào DB
            student_id = "New Student"  # Tạm thời hiển thị nhãn này
            confidence = 0.0

        # Xác định màu sắc và nhãn hiển thị
        color = (0, 255, 0) if student_id != "Unknown" else (0, 0, 255)
        label = f"ID: {student_id} ({confidence:.2f})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def capture_video(video_source=VIDEO_SOURCE):
    """
    Quản lý luồng video từ webcam hoặc file video.
    """
    try:
        # Load Haar cascade để phát hiện khuôn mặt
        face_cascade = load_haar_cascade()

        # Mở video
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise Exception(f"Cannot open video source: {video_source}")

        print("Press 'q' to quit the video stream...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame.")
                break

            # Phát hiện khuôn mặt trong frame
            faces = detect_faces(frame, face_cascade)

            # Nhận diện và hiển thị khuôn mặt
            if len(faces) > 0:
                display_faces(frame, faces)

            # Hiển thị video
            cv2.imshow("Real-Time Face Detection and Recognition", frame)

            # Thoát khi nhấn 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting video capture...")
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Real-Time Face Detection and Recognition...")
    capture_video()
