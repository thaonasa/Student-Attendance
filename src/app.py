import streamlit as st
import cv2
import numpy as np
from face_recognition import recognize_face, save_unknown_face, add_new_student, log_attendance, generate_face_embedding
from face_detection import detect_faces, load_haar_cascade
from connect_database.connect_db import DatabaseConnection
import time
import pandas as pd

def main():
    st.title("Hệ Thống Điểm Danh Bằng Nhận Diện Khuôn Mặt")
    
    # Sidebar cho các chức năng
    st.sidebar.title("Chức năng")
    app_mode = st.sidebar.selectbox(
        "Chọn chức năng:",
        ["Điểm danh", "Thêm sinh viên mới", "Xem lịch sử điểm danh"]
    )
    
    if app_mode == "Điểm danh":
        attendance_page()
    elif app_mode == "Thêm sinh viên mới":
        add_student_page()
    else:
        view_attendance_page()

def attendance_page():
    st.header("Điểm Danh")
    
    # Khởi tạo face cascade
    face_cascade = load_haar_cascade()
    
    # Tạo placeholder cho video stream
    video_placeholder = st.empty()
    
    # Tạo placeholder cho thông tin nhận diện
    info_placeholder = st.empty()
    
    # Start/Stop button
    start = st.button("Bắt đầu điểm danh")
    
    if start:
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể kết nối camera!")
                break
                
            # Phát hiện khuôn mặt
            faces = detect_faces(frame, face_cascade)
            
            for (x, y, w, h) in faces:
                cropped_face = frame[y:y+h, x:x+w]
                student_id, similarity = recognize_face(cropped_face)
                
                # Vẽ bounding box và thông tin
                if student_id:
                    color = (0, 255, 0)
                    label = f"ID: {student_id} ({similarity:.2f})"
                    log_attendance(student_id)
                    info_placeholder.success(f"Đã điểm danh: Sinh viên ID {student_id}")
                else:
                    color = (0, 0, 255)
                    label = "Unknown"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Chuyển BGR sang RGB để hiển thị trong Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame, channels="RGB")
            
            # Kiểm tra nếu người dùng muốn dừng
            if not start:
                break
                
        cap.release()

def add_student_page():
    st.header("Thêm Sinh Viên Mới")
    
    # Form nhập thông tin sinh viên
    with st.form("student_form"):
        uploaded_file = st.file_uploader("Tải lên ảnh sinh viên", type=['jpg', 'jpeg', 'png'])
        full_name = st.text_input("Họ và tên")
        date_of_birth = st.date_input("Ngày sinh")
        gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
        class_id = st.text_input("Mã lớp")
        email = st.text_input("Email")
        phone_number = st.text_input("Số điện thoại")
        
        submitted = st.form_submit_button("Thêm sinh viên")
        
        if submitted:
            if not uploaded_file:
                st.error("Vui lòng tải lên ảnh sinh viên!")
                return
                
            if not full_name or not class_id:
                st.error("Vui lòng điền đầy đủ họ tên và mã lớp!")
                return
            
            # Lưu ảnh tạm thời
            temp_image_path = f"temp_{int(time.time())}.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Thêm sinh viên vào database
                db = DatabaseConnection()
                db.connect()
                
                # Chuyển đổi date_input sang string format YYYY-MM-DD
                date_str = date_of_birth.strftime("%Y-%m-%d")
                
                # Thêm sinh viên
                db.add_student(
                    full_name, date_str, gender,
                    class_id, email, phone_number,
                    temp_image_path
                )
                
                # Tạo và thêm face embedding
                face_img = cv2.imread(temp_image_path)
                if face_img is not None:
                    face_cascade = load_haar_cascade()
                    faces = detect_faces(face_img, face_cascade)
                    
                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Lấy khuôn mặt đầu tiên
                        cropped_face = face_img[y:y+h, x:x+w]
                        embedding = generate_face_embedding(cropped_face)
                        
                        # Lấy ID sinh viên vừa thêm
                        student_id = db.cursor.lastrowid
                        db.add_face_embedding(student_id, embedding)
                        
                        st.success("Đã thêm sinh viên thành công!")
                    else:
                        st.error("Không tìm thấy khuôn mặt trong ảnh!")
                else:
                    st.error("Không thể đọc file ảnh!")
                
            except Exception as e:
                st.error(f"Lỗi khi thêm sinh viên: {str(e)}")
            finally:
                db.close()
                # Xóa file ảnh tạm
                import os
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

def view_attendance_page():
    st.header("Lịch Sử Điểm Danh")
    
    # Kết nối database
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Lấy lịch sử điểm danh
        query = """
        SELECT a.created_at, s.full_name, s.class_id, a.status
        FROM Attendance a
        JOIN Students s ON a.student_id = s.id
        ORDER BY a.created_at DESC
        """
        db.cursor.execute(query)
        attendance_records = db.cursor.fetchall()
        
        # Hiển thị dưới dạng bảng
        if attendance_records:
            st.dataframe(
                pd.DataFrame(
                    attendance_records,
                    columns=["Thời gian", "Họ tên", "Lớp", "Trạng thái"]
                )
            )
        else:
            st.info("Chưa có dữ liệu điểm danh")
            
    except Exception as e:
        st.error(f"Lỗi khi truy vấn dữ liệu: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main() 