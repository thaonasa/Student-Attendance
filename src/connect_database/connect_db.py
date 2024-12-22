import sqlite3
import datetime
import numpy as np
import os
import cv2

DATABASE_PATH = "student_attendance.db"


class DatabaseConnection:
    def __init__(self, db_name=DATABASE_PATH):
        self.db_name = db_name
        self.connection = None
        self.cursor = None

    def connect(self):
        """Kết nối đến cơ sở dữ liệu SQLite."""
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            self.cursor.execute("PRAGMA foreign_keys = ON;")
            print(f"Connected to database: {self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")

    def close(self):
        """Đóng kết nối đến cơ sở dữ liệu."""
        if self.connection:
            self.connection.close()
            print("Database connection closed.")

    def create_tables(self):
        """Tạo các bảng cần thiết nếu chưa tồn tại."""
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Students (
                    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    date_of_birth TEXT,
                    gender TEXT,
                    class_id INTEGER,
                    email TEXT,
                    phone_number TEXT,
                    photo_path TEXT
                );
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Face_Embeddings (
                    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    embedding_vector BLOB,
                    created_at TEXT,
                    FOREIGN KEY (student_id) REFERENCES Students(student_id)
                );
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Attendance (
                    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    status TEXT,
                    created_at TEXT,
                    FOREIGN KEY (student_id) REFERENCES Students(student_id)
                );
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS Attendance_History (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    entry_time TEXT,
                    exit_time TEXT,
                    status TEXT,
                    action_timestamp TEXT,
                    action_type TEXT,
                    FOREIGN KEY (student_id) REFERENCES Students(student_id)
                );
            ''')

            self.connection.commit()
            print("All tables created successfully.")
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")

    # Thêm sinh viên mới
    def add_student(self, full_name, date_of_birth, gender, class_id, email, phone_number, photo_path):
        """
        Thêm sinh viên mới và ảnh của sinh viên.
        Trả về student_id nếu thành công, None nếu thất bại.
        """
        try:
            # Kiểm tra ảnh tồn tại
            if not os.path.exists(photo_path):
                print(f"Error: Photo file not found at {photo_path}")
                return None

            # Kiểm tra ảnh có thể đọc được
            image = cv2.imread(photo_path)
            if image is None:
                print(f"Error: Cannot read image file at {photo_path}")
                return None

            # Bắt đầu transaction
            self.cursor.execute("BEGIN TRANSACTION")
            
            # Thêm sinh viên
            query = '''
                INSERT INTO Students (full_name, date_of_birth, gender, class_id, email, phone_number, photo_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            self.cursor.execute(query, (full_name, date_of_birth,
                                gender, class_id, email, phone_number, photo_path))
            
            # Lấy ID của sinh viên vừa thêm
            student_id = self.cursor.lastrowid
            
            if not student_id:
                print("Error: Failed to get student_id after insertion")
                self.cursor.execute("ROLLBACK")
                return None

            # Commit transaction
            self.connection.commit()
            print(f"New student added successfully with ID: {student_id}")
            return student_id
            
        except sqlite3.Error as e:
            print(f"Database error while adding student: {e}")
            self.cursor.execute("ROLLBACK")
            # Xóa ảnh nếu đã lưu nhưng thêm vào DB thất bại
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                    print(f"Cleaned up: Removed photo file at {photo_path}")
                except OSError as oe:
                    print(f"Error removing photo file: {oe}")
            return None
            
        except Exception as e:
            print(f"Unexpected error while adding student: {e}")
            self.cursor.execute("ROLLBACK")
            if os.path.exists(photo_path):
                try:
                    os.remove(photo_path)
                    print(f"Cleaned up: Removed photo file at {photo_path}")
                except OSError as oe:
                    print(f"Error removing photo file: {oe}")
            return None

    # Thêm vector nhúng khuôn mặt
    def add_face_embedding(self, student_id, embedding_vector):
        """
        Thêm vector nhúng khuôn mặt cho sinh viên.
        Trả về True nếu thành công, False nếu thất bại.
        """
        try:
            query = '''
                INSERT INTO Face_Embeddings (student_id, embedding_vector, created_at)
                VALUES (?, ?, ?)
            '''
            embedding_blob = embedding_vector.tobytes()  # Convert numpy array to binary
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                query, (student_id, embedding_blob, created_at))
            self.connection.commit()
            print("Face embedding added successfully.")
            return True
        except sqlite3.Error as e:
            print(f"Error adding face embedding: {e}")
            return False

    # Điểm danh sinh viên
    def mark_attendance(self, student_id, status="present"):
        """
        Ghi nhận điểm danh cho sinh viên.
        - Lưu vào Attendance_History nếu thời gian giữa 2 lần điểm danh >= 5 giây
        - Cập nhật bảng Attendance với bản ghi mới nhất
        """
        try:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Kiểm tra xem sinh viên đã điểm danh trong ngày chưa
            self.cursor.execute('''
                SELECT attendance_id, created_at, status
                FROM Attendance 
                WHERE student_id = ? AND DATE(created_at) = ?
            ''', (student_id, current_date))
            
            existing_attendance = self.cursor.fetchone()
            
            if existing_attendance:
                attendance_id, old_time, old_status = existing_attendance
                
                # Kiểm tra thời gian giữa lần điểm danh
                old_time_obj = datetime.datetime.strptime(old_time, "%Y-%m-%d %H:%M:%S")
                current_time_obj = datetime.datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
                time_diff = (current_time_obj - old_time_obj).total_seconds()
                
                # Nếu đã qua 5 giây, lưu vào history và cập nhật Attendance
                if time_diff >= 5:
                    # Lưu vào Attendance_History
                    self.cursor.execute('''
                        INSERT INTO Attendance_History 
                        (student_id, entry_time, exit_time, status, action_timestamp, action_type)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (student_id, old_time, current_time, status, current_time, 'update'))
                    
                    # Cập nhật Attendance
                    self.cursor.execute('''
                        UPDATE Attendance 
                        SET created_at = ?, status = ?
                        WHERE attendance_id = ?
                    ''', (current_time, status, attendance_id))
                    
                    print(f"Updated attendance for student {student_id} at {current_time}")
                    self.connection.commit()
                else:
                    print(f"Skipped attendance update for student {student_id} - less than 5 seconds")
            else:
                # Tạo record mới trong Attendance cho lần điểm danh đầu tiên
                self.cursor.execute('''
                    INSERT INTO Attendance 
                    (student_id, status, created_at)
                    VALUES (?, ?, ?)
                ''', (student_id, status, current_time))
                
                print(f"New attendance record for student {student_id} at {current_time}")
                self.connection.commit()
            
            return True
            
        except sqlite3.Error as e:
            print(f"Error marking attendance: {str(e)}")
            return False

    # Ghi lịch sử ra vào
    def add_attendance_history(self, student_id, entry_time, exit_time, status, action_type):
        try:
            query = '''
                INSERT INTO Attendance_History (student_id, entry_time, exit_time, status, action_timestamp, action_type)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            action_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(query, (student_id, entry_time,
                                exit_time, status, action_timestamp, action_type))
            self.connection.commit()
            print("Attendance history added successfully.")
        except sqlite3.Error as e:
            print(f"Error adding attendance history: {e}")

    # Lấy tất cả vector nhúng
    def get_all_embeddings(self):
        try:
            query = "SELECT student_id, embedding_vector FROM Face_Embeddings"
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Error fetching embeddings: {e}")
            return []

     # Cập nhật thông tin sinh viên
    def update_student(self, student_id, full_name=None, email=None, phone_number=None, photo_path=None):
        try:
            updates = []
            params = []
            if full_name:
                updates.append("full_name = ?")
                params.append(full_name)
            if email:
                updates.append("email = ?")
                params.append(email)
            if phone_number:
                updates.append("phone_number = ?")
                params.append(phone_number)
            if photo_path:
                updates.append("photo_path = ?")
                params.append(photo_path)

            if updates:
                query = f"UPDATE Students SET {', '.join(updates)} WHERE student_id = ?"
                params.append(student_id)
                self.cursor.execute(query, params)
                self.connection.commit()
                print("Student information updated successfully.")
                
                # Nếu có cập nhật ảnh mới, cập nhật lại face embedding
                if photo_path:
                    # Xóa embedding cũ
                    self.cursor.execute("DELETE FROM Face_Embeddings WHERE student_id = ?", (student_id,))
                    self.connection.commit()
                    print("Old face embedding deleted.")
                    
                    # Tạo embedding mới từ ảnh mới
                    from face_recognition import generate_face_embedding
                    import cv2
                    
                    # Đọc ảnh và tạo embedding mới
                    image = cv2.imread(photo_path)
                    if image is not None:
                        embedding_vector = generate_face_embedding(image)
                        self.add_face_embedding(student_id, embedding_vector)
                        print("New face embedding created and added.")
                    else:
                        print("Error: Could not read new photo file.")
                
                return True
        except sqlite3.Error as e:
            print(f"Error updating student: {e}")
            return False
            
    def get_student_by_id(self, student_id):
        """
        Lấy thông tin sinh viên theo ID
        """
        try:
            query = '''
                SELECT student_id, full_name, date_of_birth, gender, 
                       class_id, email, phone_number, photo_path
                FROM Students
                WHERE student_id = ?
            '''
            self.cursor.execute(query, (student_id,))
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting student: {str(e)}")
            return None

    # Xóa sinh viên và dữ liệu liên quan
    def delete_student(self, student_id):
        try:
            # Xóa vector nhúng
            self.cursor.execute(
                "DELETE FROM Face_Embeddings WHERE student_id = ?", (student_id,))
            # Xóa lịch sử điểm danh
            self.cursor.execute(
                "DELETE FROM Attendance_History WHERE student_id = ?", (student_id,))
            # Xóa trạng thái điểm danh
            self.cursor.execute(
                "DELETE FROM Attendance WHERE student_id = ?", (student_id,))
            # Xóa sinh viên
            self.cursor.execute(
                "DELETE FROM Students WHERE student_id = ?", (student_id,))
            self.connection.commit()
            print(
                f"Student with ID {student_id} and related data deleted successfully.")
        except sqlite3.Error as e:
            print(f"Error deleting student: {e}")

    # Truy vấn lịch sử điểm danh theo ngày
    def get_attendance_by_date(self, date):
        try:
            query = '''
                SELECT Students.full_name, Attendance.status, Attendance.created_at
                FROM Attendance
                JOIN Students ON Attendance.student_id = Students.student_id
                WHERE DATE(Attendance.created_at) = ?
            '''
            self.cursor.execute(query, (date,))
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Error fetching attendance: {e}")
            return []

    # Truy vấn lịch sử ra/vào của sinh viên
    def get_attendance_history(self, student_id, date=None):
        """
        Truy vấn lịch sử điểm danh của sinh viên.
        :param student_id: ID của sinh viên
        :param date: Ngày cần xem lịch sử (tùy chọn)
        :return: Danh sách các bản ghi lịch sử điểm danh
        """
        try:
            if date:
                query = '''
                    SELECT created_at, status, action_type
                    FROM Attendance_History
                    WHERE student_id = ? AND DATE(created_at) = ?
                    ORDER BY created_at DESC
                '''
                self.cursor.execute(query, (student_id, date))
            else:
                query = '''
                    SELECT created_at, status, action_type
                    FROM Attendance_History
                    WHERE student_id = ?
                    ORDER BY created_at DESC
                '''
                self.cursor.execute(query, (student_id,))
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching attendance history: {str(e)}")
            return []

    def get_all_students(self, class_id=None):
        """
        Lấy danh sách tất cả sinh viên hoặc sinh viên của một lớp cụ thể.
        :param class_id: Mã lớp cần lọc (tùy chọn)
        :return: Danh sách sinh viên
        """
        try:
            if class_id:
                query = '''
                    SELECT student_id, full_name, date_of_birth, gender, 
                           class_id, email, phone_number, photo_path
                    FROM Students
                    WHERE class_id = ?
                    ORDER BY full_name
                '''
                self.cursor.execute(query, (class_id,))
            else:
                query = '''
                    SELECT student_id, full_name, date_of_birth, gender, 
                           class_id, email, phone_number, photo_path
                    FROM Students
                    ORDER BY full_name
                '''
                self.cursor.execute(query)
            
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting students: {str(e)}")
            return []
