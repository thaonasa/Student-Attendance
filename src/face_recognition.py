import cv2
import os
import numpy as np
from keras.models import load_model
from face_detection import detect_faces, load_haar_cascade
from connect_database.connect_db import DatabaseConnection

# Đường dẫn lưu ảnh chưa nhận diện
UNKNOWN_FACE_DIR = "data/unknown_faces/"

# Đường dẫn đến mô hình nhận diện khuôn mặt
FINAL_MODEL_PATH = "models/final_model/final_model.h5"

# Đảm bảo thư mục lưu ảnh chưa nhận diện tồn tại
os.makedirs(UNKNOWN_FACE_DIR, exist_ok=True)

# Tải mô hình nhận diện khuôn mặt
model = load_model(FINAL_MODEL_PATH)

# Hàm để chuyển embedding từ bytes (trong SQLite) sang numpy array


def bytes_to_vector(blob):
    return np.frombuffer(blob, dtype=np.float32)

# Tính toán độ tương đồng Cosine


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def log_attendance(student_id):
    db = DatabaseConnection()
    db.connect()
    try:
        query = '''
            INSERT INTO Attendance (student_id, status, created_at)
            VALUES (?, 'Present', datetime('now'))
        '''
        db.cursor.execute(query, (student_id,))
        db.connection.commit()
        print(f"Attendance logged for student ID: {student_id}")
    except Exception as e:
        print(f"Error logging attendance: {e}")
    finally:
        db.close()

# Nhận diện khuôn mặt bằng cách so sánh vector nhúng


def recognize_face(cropped_face):
    """
    Nhận diện khuôn mặt bằng cách so sánh vector nhúng với cơ sở dữ liệu.
    :param cropped_face: Ảnh khuôn mặt đã cắt (numpy array).
    :return: (student_id, similarity) hoặc (None, 0.0)
    """
    db = DatabaseConnection()
    db.connect()

    try:
        # Tạo vector nhúng cho khuôn mặt
        face_embedding = generate_face_embedding(cropped_face)

        # Lấy tất cả vector nhúng từ cơ sở dữ liệu
        embeddings = db.get_all_embeddings()
        best_match = None
        max_similarity = 0.0

        for student_id, db_embedding in embeddings:
            db_vector = np.frombuffer(db_embedding, dtype=np.float32)
            similarity = cosine_similarity(face_embedding, db_vector)
            if similarity > max_similarity and similarity > 0.8:  # Ngưỡng tương đồng
                best_match = student_id
                max_similarity = similarity

        return best_match, max_similarity
    finally:
        db.close()


def save_unknown_face(cropped_face):
    """
    Lưu ảnh khuôn mặt chưa nhận diện vào thư mục tạm.
    :param cropped_face: Ảnh khuôn mặt đã cắt (numpy array).
    :return: Đường dẫn ảnh đã lưu.
    """
    image_name = f"unknown_{len(os.listdir(UNKNOWN_FACE_DIR)) + 1}.jpg"
    image_path = os.path.join(UNKNOWN_FACE_DIR, image_name)
    cv2.imwrite(image_path, cropped_face)
    print(f"Unknown face saved at: {image_path}")
    return image_path


def add_new_student(image_path):
    """
    Thêm sinh viên mới vào cơ sở dữ liệu.
    :param image_path: Đường dẫn ảnh khuôn mặt sinh viên.
    """
    print("New student detected! Please enter the following details:")
    full_name = input("Full Name: ")
    date_of_birth = input("Date of Birth (YYYY-MM-DD): ")
    gender = input("Gender (Male/Female): ")
    class_id = input("Class ID: ")
    email = input("Email: ")
    phone_number = input("Phone Number: ")

    # Lưu thông tin sinh viên vào cơ sở dữ liệu
    db = DatabaseConnection()
    db.connect()
    db.add_student(full_name, date_of_birth, gender,
                   class_id, email, phone_number, image_path)

    # Lấy ID sinh viên vừa thêm
    student_id = db.cursor.lastrowid

    # Tạo vector nhúng và lưu vào cơ sở dữ liệu
    cropped_face = cv2.imread(image_path)
    embedding_vector = generate_face_embedding(cropped_face)
    db.add_face_embedding(student_id, embedding_vector)

    print(f"New student {full_name} added successfully with ID {student_id}.")
    db.close()


def generate_face_embedding(cropped_face):
    """
    Trích xuất vector nhúng từ ảnh khuôn mặt.
    :param cropped_face: Ảnh khuôn mặt đã cắt (numpy array).
    :return: Vector nhúng (numpy array).
    """
    resized_face = cv2.resize(cropped_face, (160, 160))
    normalized_face = resized_face / 255.0
    input_face = np.expand_dims(normalized_face, axis=0)
    embedding = model.predict(input_face)[0]
    return embedding


def main():
    # Kết nối cơ sở dữ liệu
    db = DatabaseConnection()
    db.connect()

    # Khởi tạo Haar Cascade
    face_cascade = load_haar_cascade()

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        db.close()
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        # Phát hiện khuôn mặt
        faces = detect_faces(frame, face_cascade)

        for (x, y, w, h) in faces:
            cropped_face = frame[y:y+h, x:x+w]
            student_id, similarity = recognize_face(cropped_face)  # Đã sửa lại cách gọi hàm

            # Vẽ bounding box và ID sinh viên
            if student_id:
                log_attendance(student_id)  # Ghi nhận điểm danh
                color = (0, 255, 0)
                label = f"ID: {student_id} ({similarity:.2f})"
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị video
        cv2.imshow("Face Recognition and Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()
