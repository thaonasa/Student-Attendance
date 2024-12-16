from connect_database.connect_db import DatabaseConnection


def add_student(full_name, date_of_birth, gender, class_id, email, phone_number, photo_path):
    db = DatabaseConnection()
    db.connect()
    try:
        query = '''
            INSERT INTO Students (full_name, date_of_birth, gender, class_id, email, phone_number, photo_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        db.cursor.execute(query, (full_name, date_of_birth,
                          gender, class_id, email, phone_number, photo_path))
        db.connection.commit()
        print("New student added successfully.")
    except Exception as e:
        print(f"Error adding student: {e}")
    finally:
        db.close()


def delete_student(student_id):
    db = DatabaseConnection()
    db.connect()
    try:
        # Xóa vector nhúng trước
        db.cursor.execute(
            "DELETE FROM Face_Embeddings WHERE student_id = ?", (student_id,))
        # Xóa sinh viên
        db.cursor.execute(
            "DELETE FROM Students WHERE student_id = ?", (student_id,))
        db.connection.commit()
        print("Student deleted successfully.")
    except Exception as e:
        print(f"Error deleting student: {e}")
    finally:
        db.close()
