import sqlite3
import datetime
import numpy as np

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
        try:
            query = '''
                INSERT INTO Students (full_name, date_of_birth, gender, class_id, email, phone_number, photo_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            self.cursor.execute(query, (full_name, date_of_birth,
                                gender, class_id, email, phone_number, photo_path))
            self.connection.commit()
            print("New student added successfully.")
        except sqlite3.Error as e:
            print(f"Error adding student: {e}")

    # Thêm vector nhúng khuôn mặt
    def add_face_embedding(self, student_id, embedding_vector):
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
        except sqlite3.Error as e:
            print(f"Error adding face embedding: {e}")

    # Điểm danh sinh viên
    def mark_attendance(self, student_id, status):
        try:
            query = '''
                INSERT INTO Attendance (student_id, status, created_at)
                VALUES (?, ?, ?)
            '''
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(query, (student_id, status, created_at))
            self.connection.commit()
            print("Attendance marked successfully.")
        except sqlite3.Error as e:
            print(f"Error marking attendance: {e}")

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

    def update_student(self, student_id, full_name=None, email=None, phone_number=None):
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

            if updates:
                query = f"UPDATE Students SET {', '.join(updates)} WHERE student_id = ?"
                params.append(student_id)
                self.cursor.execute(query, params)
                self.connection.commit()
                print("Student information updated successfully.")
        except sqlite3.Error as e:
            print(f"Error updating student: {e}")

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
    def get_attendance_history(self, student_id):
        try:
            query = '''
                SELECT entry_time, exit_time, status, action_timestamp, action_type
                FROM Attendance_History
                WHERE student_id = ?
            '''
            self.cursor.execute(query, (student_id,))
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Error fetching attendance history: {e}")
            return []
