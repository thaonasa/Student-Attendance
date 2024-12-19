import streamlit as st
import cv2
import numpy as np
from face_recognition import recognize_face, save_unknown_face, add_new_student, log_attendance, generate_face_embedding
from face_detection import detect_faces, load_haar_cascade
from connect_database.connect_db import DatabaseConnection
import time
import pandas as pd
from datetime import datetime
import io
import re

def main():
    st.title("Hệ Thống Điểm Danh Bằng Nhận Diện Khuôn Mặt")
    
    st.sidebar.title("Chức năng")
    app_mode = st.sidebar.selectbox(
        "Chọn chức năng:",
        ["Điểm danh", "Thêm sinh viên mới", "Quản lý sinh viên", "Xem & Xuất điểm danh"]
    )
    
    if app_mode == "Điểm danh":
        attendance_page()
    elif app_mode == "Thêm sinh viên mới":
        add_student_page()
    elif app_mode == "Quản lý sinh viên":
        manage_students_page()
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
        
        # Validation cho họ tên
        full_name = st.text_input(
            "Họ và tên",
            help="Họ tên phải có ít nhất 2 từ và không chứa số"
        )
        
        # Giới hạn ngày sinh: không cho phép chọn ngày sau ngày hiện tại
        max_date = datetime.now().date()
        min_date = datetime.now().date().replace(year=datetime.now().year - 100)
        date_of_birth = st.date_input(
            "Ngày sinh",
            min_value=min_date,
            max_value=max_date,
            value=datetime.now().date(),
            help="Ngày sinh không được sau ngày hiện tại"
        )
        
        gender = st.selectbox("Giới tính", ["Nam", "Nữ"])
        
        # Validation cho mã lớp
        class_id = st.text_input(
            "Mã lớp",
            help="Mã lớp phải có định dạng: XXXXnnn (X: chữ cái, n: số)"
        )
        
        # Validation cho email
        email = st.text_input(
            "Email",
            help="Email phải có định dạng hợp lệ (vd: example@domain.com)"
        )
        
        # Validation cho số điện thoại
        phone_number = st.text_input(
            "Số điện thoại",
            help="Số điện thoại phải có 10 chữ số và bắt đầu bằng 0"
        )
        
        submitted = st.form_submit_button("Thêm sinh viên")
        
        if submitted:
            # Kiểm tra ảnh
            if not uploaded_file:
                st.error("Vui lòng tải lên ảnh sinh viên!")
                return
            
            # Kiểm tra họ tên
            if not full_name or len(full_name.split()) < 2 or any(char.isdigit() for char in full_name):
                st.error("Họ tên không hợp lệ! Vui lòng nhập đầy đủ họ tên và không chứa số.")
                return
            
            # Kiểm tra mã lớp
            if not re.match(r'^[A-Za-z]{4}\d{3}$', class_id):
                st.error("Mã lớp không hợp lệ! Phải có định dạng XXXXnnn.")
                return
            
            # Kiểm tra email
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                st.error("Email không hợp lệ!")
                return
            
            # Kiểm tra số điện thoại
            if not re.match(r'^0\d{9}$', phone_number):
                st.error("Số điện thoại không hợp lệ! Phải có 10 chữ số và bắt đầu bằng 0.")
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
                
                db.add_student(
                    full_name, date_str, gender,
                    class_id, email, phone_number,
                    temp_image_path
                )
                
                st.success("Đã thêm sinh viên thành công!")
                
            except Exception as e:
                st.error(f"Lỗi khi thêm sinh viên: {str(e)}")
            finally:
                db.close()
                # Xóa file ảnh tạm
                import os
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)

def manage_students_page():
    st.header("Quản Lý Sinh Viên")
    
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Lấy danh sách các lớp
        db.cursor.execute("SELECT DISTINCT class_id FROM Students")
        classes = [row[0] for row in db.cursor.fetchall()]
        
        # Chọn lớp để xem
        selected_class = st.selectbox("Chọn lớp:", ["Tất cả"] + classes)
        
        # Lấy danh sách sinh viên
        if selected_class == "Tất cả":
            query = """
            SELECT student_id, full_name, date_of_birth, gender, class_id, email, phone_number 
            FROM Students
            """
            db.cursor.execute(query)
        else:
            query = """
            SELECT student_id, full_name, date_of_birth, gender, class_id, email, phone_number 
            FROM Students WHERE class_id = ?
            """
            db.cursor.execute(query, (selected_class,))
            
        students = db.cursor.fetchall()
        
        if students:
            df = pd.DataFrame(students, columns=[
                "ID", "Họ tên", "Ngày sinh", "Giới tính", 
                "Lớp", "Email", "Số điện thoại"
            ])
            st.dataframe(df)
            
            # Nút xóa sinh viên
            if st.button("Xóa sinh viên đã chọn"):
                selected_rows = st.multiselect(
                    "Chọn sinh viên cần xóa:", 
                    df["Họ tên"].tolist()
                )
                if selected_rows:
                    for name in selected_rows:
                        db.cursor.execute(
                            "DELETE FROM Students WHERE full_name = ?", 
                            (name,)
                        )
                    db.connection.commit()
                    st.success("Đã xóa sinh viên thành công!")
                    st.rerun()
        else:
            st.info("Không có sinh viên trong lớp này")
            
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
    finally:
        db.close()

def view_attendance_page():
    st.header("Xem & Xuất Điểm Danh")
    
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Lấy danh sách lớp
        db.cursor.execute("SELECT DISTINCT class_id FROM Students")
        classes = [row[0] for row in db.cursor.fetchall()]
        
        # Chọn lớp và ngày
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox("Chọn lớp:", ["Tất cả"] + classes)
        with col2:
            selected_date = st.date_input("Chọn ngày:", datetime.now())
        
        # Query điểm danh
        if selected_class == "Tất cả":
            query = """
            SELECT a.created_at, s.full_name, s.class_id, a.status
            FROM Attendance a
            JOIN Students s ON a.student_id = s.student_id
            WHERE DATE(a.created_at) = ?
            ORDER BY a.created_at DESC
            """
            params = (selected_date.strftime("%Y-%m-%d"),)
        else:
            query = """
            SELECT a.created_at, s.full_name, s.class_id, a.status
            FROM Attendance a
            JOIN Students s ON a.student_id = s.student_id
            WHERE s.class_id = ? AND DATE(a.created_at) = ?
            ORDER BY a.created_at DESC
            """
            params = (selected_class, selected_date.strftime("%Y-%m-%d"))
            
        db.cursor.execute(query, params)
        attendance_records = db.cursor.fetchall()
        
        if attendance_records:
            df = pd.DataFrame(
                attendance_records,
                columns=["Thời gian", "Họ tên", "Lớp", "Trạng thái"]
            )
            st.dataframe(df)
            
            # Xuất file Excel
            if st.button("Xuất Excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Điểm danh', index=False)
                
                excel_data = output.getvalue()
                file_name = f"diem_danh_{selected_class}_{selected_date}.xlsx"
                
                st.download_button(
                    label="Tải file Excel",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("Không có dữ liệu điểm danh cho ngày đã chọn")
            
    except Exception as e:
        st.error(f"Lỗi khi truy vấn dữ liệu: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main() 