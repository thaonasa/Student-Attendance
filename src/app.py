import os
import pytz
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
from rembg import remove
from PIL import Image
from pathlib import Path

# Tạo thư mục lưu ảnh nếu chưa tồn tại
STUDENT_IMAGES_DIR = "student_images"
Path(STUDENT_IMAGES_DIR).mkdir(parents=True, exist_ok=True)

def main():
    st.title("Hệ Thống Điểm Danh Bằng Nhận Diện Khuôn Mặt")
    
    st.sidebar.title("Chức năng")
    app_mode = st.sidebar.selectbox(
        "Chọn chức năng:",
        ["Điểm danh", "Thêm sinh viên mới", "Quản lý sinh viên", "Xem & Xuất điểm danh", "Lịch sử ra/vào"]
    )
    
    if app_mode == "Điểm danh":
        attendance_page()
    elif app_mode == "Thêm sinh viên mới":
        add_student_page()
    elif app_mode == "Quản lý sinh viên":
        manage_students_page()
    elif app_mode == "Lịch sử ra/vào":
        attendance_history_page()
    else:
        view_attendance_page()

def attendance_page():
    st.header("Điểm Danh")
    
    # Khởi tạo database connection
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Lấy danh sách các lớp
        db.cursor.execute("SELECT DISTINCT class_id FROM Students")
        classes = [row[0] for row in db.cursor.fetchall()]
        
        if not classes:
            st.warning("Chưa có lớp nào trong hệ thống. Vui lòng thêm sinh viên trước!")
            return
        
        # Chọn lớp để điểm danh
        selected_class = st.selectbox("Chọn lớp cần điểm danh:", classes)
        
        # Khởi tạo face cascade
        face_cascade = load_haar_cascade()
        
        # Tạo placeholder cho video stream
        video_placeholder = st.empty()
        
        # Tạo placeholder cho thông tin nhận diện
        info_placeholder = st.empty()
        
        # Start/Stop button
        col1, col2 = st.columns(2)
        with col1:
            start = st.button("Bắt đầu điểm danh")
        with col2:
            stop = st.button("Dừng điểm danh")
        
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        if start:
            # Đặt trạng thái absent cho tất cả sinh viên trong lớp
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Lấy danh sách sinh viên trong lớp
            db.cursor.execute(
                "SELECT student_id FROM Students WHERE class_id = ?",
                (selected_class,)
            )
            students = db.cursor.fetchall()
            
            # Đặt trạng thái absent cho từng sinh viên
            for student in students:
                # Kiểm tra xem sinh viên đã có điểm danh hôm nay chưa
                db.cursor.execute('''
                    SELECT attendance_id FROM Attendance 
                    WHERE student_id = ? AND DATE(created_at) = ?
                ''', (student[0], current_date))
                
                existing_record = db.cursor.fetchone()
                
                if existing_record:
                    # Nếu đã có record, cập nhật thành absent
                    db.cursor.execute('''
                        UPDATE Attendance 
                        SET status = 'absent', created_at = ?
                        WHERE attendance_id = ?
                    ''', (current_time, existing_record[0]))
                else:
                    # Nếu chưa có record, tạo mới với status absent
                    db.cursor.execute('''
                        INSERT INTO Attendance (student_id, status, created_at)
                        VALUES (?, 'absent', ?)
                    ''', (student[0], current_time))
            
            db.connection.commit()
            st.success(f"Đã đặt trạng thái absent cho tất cả sinh viên lớp {selected_class}")
            
            st.session_state.camera_running = True
            cap = cv2.VideoCapture(0)
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Không thể kết nối camera!")
                    break
                    
                # Phát hiện khuôn mặt
                faces = detect_faces(frame, face_cascade)
                
                for (x, y, w, h) in faces:
                    cropped_face = frame[y:y+h, x:x+w]
                    student_id, similarity = recognize_face(cropped_face)
                    
                    # Kiểm tra xem sinh viên có thuộc lớp đang điểm danh không
                    if student_id:
                        db.cursor.execute(
                            "SELECT class_id, full_name FROM Students WHERE student_id = ?",
                            (student_id,)
                        )
                        student_info = db.cursor.fetchone()
                        if student_info and student_info[0] == selected_class:
                            # Sinh viên thuộc lớp đang điểm danh
                            color = (0, 255, 0)  # Màu xanh lá
                            student_class = student_info[0]
                            student_name = student_info[1]
                            # Loại bỏ dấu trong tên hiển thị
                            label = f"{student_name} - {similarity:.2f}"
                            log_attendance(student_id, "present")  # Cập nhật trạng thái thành present
                            info_placeholder.success(f"Da diem danh: {student_name} - Lop {student_class}")
                        else:
                            # Sinh viên không thuộc lớp đang điểm danh
                            color = (0, 165, 255)  # Màu cam
                            label = f"Khong thuoc lop {selected_class}"
                    else:
                        # Không nhận diện được sinh viên
                        color = (0, 0, 255)  # Màu đỏ
                        label = "Khong nhan dien"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Chuyển BGR sang RGB để hiển thị trong Streamlit
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame, channels="RGB")
                
            cap.release()
        
        if stop:
            st.session_state.camera_running = False
            st.rerun()
        
        # Hiển thị danh sách điểm danh của lớp hôm nay
        st.write(f"### Danh sách điểm danh lớp {selected_class} hôm nay")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Truy vấn trực tiếp để lấy thông tin điểm danh
        query = '''
            SELECT s.full_name, a.status, a.created_at
            FROM Students s
            JOIN Attendance a ON s.student_id = a.student_id
            WHERE s.class_id = ? AND DATE(a.created_at) = ?
            ORDER BY a.created_at DESC
        '''
        db.cursor.execute(query, (selected_class, current_date))
        attendance_records = db.cursor.fetchall()
        
        if attendance_records:
            for record in attendance_records:
                status_color = "green" if record[1] == "present" else "red"
                st.markdown(
                    f"- {record[0]}: <span style='color: {status_color}'>{record[1]}</span> ({record[2]})",
                    unsafe_allow_html=True
                )
        else:
            st.info(f"Chưa có sinh viên lớp {selected_class} điểm danh hôm nay")
            
    except Exception as e:
        st.error(f"Lỗi: {str(e)}")
    finally:
        db.close()

def add_student_page():
    st.header("Thêm Sinh Viên Mới")
    
    # Chọn phương thức thêm ảnh
    image_source = st.radio("Chọn cách thêm ảnh:", ["Tải ảnh lên", "Chụp ảnh"])
    
    captured_image = None
    uploaded_file = None
    processed_image = None
    
    if image_source == "Chụp ảnh":
        st.write("Camera")
        camera_placeholder = st.empty()
        capture_button = st.button("Chụp ảnh")
        
        if not hasattr(st.session_state, 'camera'):
            st.session_state.camera = cv2.VideoCapture(0)
        
        ret, frame = st.session_state.camera.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame, channels="RGB")
            
            if capture_button:
                captured_image = frame
                processed_image = frame.copy()
                st.session_state.camera.release()
                delattr(st.session_state, 'camera')
                st.success("Đã chụp ảnh thành công!")
                st.image(captured_image, caption="Ảnh đã chụp")
    else:
        uploaded_file = st.file_uploader("Tải lên ảnh sinh viên", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            image = Image.open(uploaded_file)
            processed_image = np.array(image)
            st.image(image, caption="Ảnh đã tải lên")
    
    # Tùy chọn xóa background
    remove_bg = st.checkbox("Xóa background")
    
    if (uploaded_file or captured_image) and remove_bg:
        try:
            if uploaded_file:
                input_image = Image.open(uploaded_file)
            else:
                input_image = Image.fromarray(captured_image)
            
            # Xóa background
            output_image = remove(input_image)
            st.image(output_image, caption="Ảnh đã xóa background")
            processed_image = np.array(output_image)
            
        except Exception as e:
            st.error(f"Lỗi khi xóa background: {str(e)}")
    
    # Form nhập thông tin sinh viên
    with st.form("student_form"):
        full_name = st.text_input("Họ và tên")
        
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
        class_id = st.text_input("Mã lớp")
        email = st.text_input("Email", help="Email phải có định dạng hợp lệ (vd: example@domain.com)")
        phone_number = st.text_input("Số điện thoại", help="Số điện thoại phải có 10 chữ số và bắt đầu bằng 0")
        
        submitted = st.form_submit_button("Thêm sinh viên")
        
        if submitted:
            # Kiểm tra ảnh
            if processed_image is None:
                st.error("Vui lòng tải lên ảnh hoặc chụp ảnh sinh viên!")
                return
            
            # Kiểm tra các trường bắt buộc
            if not full_name:
                st.error("Vui lòng nhập họ tên sinh viên!")
                return
            
            if not class_id:
                st.error("Vui lòng nhập mã lớp!")
                return
            
            # Kiểm tra các trường có định dạng đặc biệt
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                st.error("Email không hợp lệ!")
                return
            
            if not re.match(r'^0\d{9}$', phone_number):
                st.error("Số điện thoại không hợp lệ! Phải có 10 chữ số và bắt đầu bằng 0.")
                return
            
            try:
                # Tạo tên file ảnh duy nhất
                timestamp = int(time.time())
                image_filename = f"{timestamp}_{class_id}.jpg"
                image_path = os.path.join(STUDENT_IMAGES_DIR, image_filename)
                
                # Chuyển đổi ảnh sang BGR trước khi lưu
                if len(processed_image.shape) == 3:
                    if processed_image.shape[2] == 4:
                        # Nếu ảnh có kênh alpha (RGBA), chuyển sang BGR
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGBA2BGR)
                    elif processed_image.shape[2] == 3:
                        # Nếu ảnh RGB, chuyển sang BGR
                        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                
                # Lưu ảnh
                cv2.imwrite(image_path, processed_image)
                
                if not os.path.exists(image_path):
                    st.error("Lỗi: Không thể lưu file ảnh!")
                    return
                
                # Phát hiện khuôn mặt trước khi thêm vào database
                face_cascade = load_haar_cascade()
                face_img = cv2.imread(image_path)
                faces = detect_faces(face_img, face_cascade)
                
                # Kiểm tra faces có phải là list rỗng hoặc None
                if faces is None or len(faces) == 0:
                    st.error("Không tìm thấy khuôn mặt trong ảnh!")
                    os.remove(image_path)
                    return
                
                # Kết nối database và thêm sinh viên
                db = DatabaseConnection()
                db.connect()
                
                try:
                    # Thêm sinh viên và lấy ID
                    student_id = db.add_student(
                        full_name, 
                        date_of_birth.strftime("%Y-%m-%d"), 
                        gender,
                        class_id, 
                        email, 
                        phone_number,
                        image_path
                    )
                    
                    if student_id:
                        # Cắt và tạo embedding từ khuôn mặt đã phát hiện
                        x, y, w, h = faces[0]  # Lấy khuôn mặt đầu tiên
                        cropped_face = face_img[y:y+h, x:x+w]
                        
                        try:
                            # Tạo embedding
                            embedding = generate_face_embedding(cropped_face)
                            if embedding is None:
                                st.error("Lỗi khi tạo face embedding! Vui lòng thử lại với ảnh khác.")
                                db.delete_student(student_id)
                                os.remove(image_path)
                                return
                            
                            # Thêm face embedding vào database
                            if db.add_face_embedding(student_id, embedding):
                                st.success("Đã thêm sinh viên thành công!")
                                st.success(f"Ảnh được lưu tại: {image_path}")
                                time.sleep(2)  # Đợi 2 giây để người dùng thấy thông báo
                                st.experimental_rerun()
                            else:
                                st.error("Lỗi khi lưu face embedding vào database! Vui lòng thử lại.")
                                db.delete_student(student_id)
                                os.remove(image_path)
                                
                        except Exception as e:
                            st.error(f"Lỗi khi xử lý face embedding: {str(e)}")
                            st.error("Vui lòng thử lại với ảnh khác.")
                            db.delete_student(student_id)
                            os.remove(image_path)
                    else:
                        st.error("Lỗi khi thêm thông tin sinh viên!")
                        os.remove(image_path)
                        
                finally:
                    db.close()
                    
            except Exception as e:
                st.error(f"Lỗi khi thêm sinh viên: {str(e)}")
                # Xóa ảnh nếu có lỗi và file tồn tại
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
                if 'db' in locals():
                    db.close()

def reset_database():
    try:
        db = DatabaseConnection()
        db.connect()
        
        # Xóa tất cả dữ liệu từ các bảng
        db.cursor.execute("DELETE FROM Attendance")
        db.cursor.execute("DELETE FROM FaceEmbeddings")
        db.cursor.execute("DELETE FROM Students")
        db.connection.commit()
        
        # Xóa tất cả ảnh trong thư mục student_images
        for file in os.listdir(STUDENT_IMAGES_DIR):
            file_path = os.path.join(STUDENT_IMAGES_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return True, "Đã reset database và xóa tất cả ảnh thành công!"
    except Exception as e:
        return False, f"Lỗi khi reset database: {str(e)}"
    finally:
        db.close()

def manage_students_page():
    st.title("Quản lý sinh viên")
    
    # Kết nối database
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Lấy danh sách các lớp
        db.cursor.execute("SELECT DISTINCT class_id FROM Students ORDER BY class_id")
        classes = [row[0] for row in db.cursor.fetchall()]
        
        if not classes:
            st.warning("Không có sinh viên nào trong cơ sở dữ liệu.")
            return
            
        # Chọn lớp để xem và quản lý
        selected_class = st.selectbox(
            "Chọn lớp:",
            ["Tất cả"] + classes
        )
        
        # Lấy danh sách sinh viên theo lớp
        if selected_class == "Tất cả":
            students = db.get_all_students()
        else:
            students = db.get_all_students(class_id=selected_class)
        
        if not students:
            st.warning(f"Không có sinh viên nào trong lớp {selected_class}.")
            return
            
        # Tạo DataFrame từ danh sách sinh viên
        df = pd.DataFrame(students, columns=[
            'ID', 'Họ tên', 'Ngày sinh', 'Giới tính', 
            'Mã lớp', 'Email', 'Số điện thoại', 'Ảnh'
        ])
        
        # Hiển thị thống kê
        st.subheader("Thống kê")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Tổng số sinh viên: {len(df)}")
        with col2:
            if selected_class != "Tất cả":
                st.info(f"Lớp: {selected_class}")
        
        # Hiển thị danh sách sinh viên
        st.subheader("Danh sách sinh viên")
        st.dataframe(df)
        
        # Chọn sinh viên để chỉnh sửa
        selected_student_id = st.selectbox(
            "Chọn sinh viên để chỉnh sửa:",
            options=df['ID'].tolist(),
            format_func=lambda x: f"{x} - {df[df['ID'] == x]['Họ tên'].iloc[0]} ({df[df['ID'] == x]['Mã lớp'].iloc[0]})"
        )
        
        if selected_student_id:
            student = db.get_student_by_id(selected_student_id)
            if student:
                st.subheader("Chỉnh sửa thông tin sinh viên")
                
                # Hiển thị ảnh hiện tại
                col1, col2 = st.columns([1, 2])
                with col1:
                    if student[7]:  # photo_path
                        st.subheader("Ảnh hiện tại")
                        try:
                            image = Image.open(student[7])
                            st.image(image, width=200)
                        except Exception as e:
                            st.error(f"Không thể hiển thị ảnh: {str(e)}")
                
                with col2:
                    # Form chỉnh sửa thông tin
                    with st.form("edit_student_form"):
                        new_name = st.text_input("Họ tên:", value=student[1])
                        new_email = st.text_input("Email:", value=student[5])
                        new_phone = st.text_input("Số điện thoại:", value=student[6])
                        
                        # Thêm phần upload ảnh mới
                        new_photo = st.file_uploader("Cập nhật ảnh mới:", type=['jpg', 'jpeg', 'png'])
                        
                        submit_button = st.form_submit_button("Cập nhật")
                        
                        if submit_button:
                            # Xử lý ảnh mới nếu được upload
                            new_photo_path = None
                            if new_photo:
                                # Tạo tên file mới với timestamp
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                new_filename = f"student_{selected_student_id}_{timestamp}.jpg"
                                new_photo_path = os.path.join("student_images", new_filename)
                                
                                # Lưu ảnh mới
                                with open(new_photo_path, "wb") as f:
                                    f.write(new_photo.getbuffer())
                            
                            # Cập nhật thông tin sinh viên
                            success = db.update_student(
                                selected_student_id,
                                full_name=new_name,
                                email=new_email,
                                phone_number=new_phone,
                                photo_path=new_photo_path if new_photo else None
                            )
                            
                            if success:
                                st.success("Cập nhật thông tin sinh viên thành công!")
                                if new_photo:
                                    st.success("Ảnh mới đã được cập nhật và face embedding đã được tạo lại!")
                                st.experimental_rerun()
                            else:
                                st.error("Có lỗi xảy ra khi cập nhật thông tin sinh viên!")
                
                # Nút xóa sinh viên
                if st.button("Xóa sinh viên này", type="primary"):
                    if st.warning("Bạn có chắc chắn muốn xóa sinh viên này? Hành động này không thể hoàn tác!"):
                        db.delete_student(selected_student_id)
                        st.success("Đã xóa sinh viên thành công!")
                        st.experimental_rerun()
    
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
            selected_date = st.date_input("Chọn ngày:", datetime.now().astimezone(pytz.timezone('Asia/Ho_Chi_Minh')))
        
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

def attendance_history_page():
    st.header("Lịch sử ra/vào")
    
    db = DatabaseConnection()
    db.connect()
    
    try:
        # Lấy danh sách lớp
        db.cursor.execute("SELECT DISTINCT class_id FROM Students ORDER BY class_id")
        classes = [row[0] for row in db.cursor.fetchall()]
        
        # Chọn lớp và ngày
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox("Chọn lớp:", ["Tất cả"] + classes)
        with col2:
            selected_date = st.date_input("Chọn ngày:", datetime.now().astimezone(pytz.timezone('Asia/Ho_Chi_Minh')))
        
        # Query lịch sử ra/vào
        if selected_class == "Tất cả":
            query = """
            SELECT 
                s.full_name,
                s.class_id,
                ah.entry_time,
                ah.exit_time,
                ah.status,
                ah.action_type
            FROM Attendance_History ah
            JOIN Students s ON ah.student_id = s.student_id
            WHERE DATE(ah.action_timestamp) = ?
            ORDER BY ah.action_timestamp DESC
            """
            params = (selected_date.strftime("%Y-%m-%d"),)
        else:
            query = """
            SELECT 
                s.full_name,
                s.class_id,
                ah.entry_time,
                ah.exit_time,
                ah.status,
                ah.action_type
            FROM Attendance_History ah
            JOIN Students s ON ah.student_id = s.student_id
            WHERE s.class_id = ? AND DATE(ah.action_timestamp) = ?
            ORDER BY ah.action_timestamp DESC
            """
            params = (selected_class, selected_date.strftime("%Y-%m-%d"))
        
        db.cursor.execute(query, params)
        history_records = db.cursor.fetchall()
        
        if history_records:
            # Tạo DataFrame
            df = pd.DataFrame(
                history_records,
                columns=["Họ tên", "Lớp", "Thời gian vào", "Thời gian ra", "Trạng thái", "Loại"]
            )
            
            # Hiển thị thống kê
            st.subheader("Thống kê")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"Tổng số bản ghi: {len(df)}")
            with col2:
                if selected_class != "Tất cả":
                    st.info(f"Lớp: {selected_class}")
            with col3:
                st.info(f"Ngày: {selected_date.strftime('%d/%m/%Y')}")
            
            # Hiển thị dữ liệu
            st.subheader("Danh sách chi tiết")
            st.dataframe(df)
            
            # Xuất Excel
            if st.button("Xuất Excel"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Lịch sử ra vào', index=False)
                
                excel_data = output.getvalue()
                file_name = f"lich_su_ra_vao_{selected_class}_{selected_date.strftime('%Y%m%d')}.xlsx"
                
                st.download_button(
                    label="Tải file Excel",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("Không có dữ liệu lịch sử ra/vào cho ngày đã chọn")
            
    except Exception as e:
        st.error(f"Lỗi khi truy vấn dữ liệu: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    main() 