from connect_database.connect_db import DatabaseConnection


def get_attendance_by_date(date):
    db = DatabaseConnection()
    db.connect()
    try:
        query = '''
            SELECT Students.full_name, Attendance.status, Attendance.created_at
            FROM Attendance
            JOIN Students ON Attendance.student_id = Students.student_id
            WHERE DATE(Attendance.created_at) = ?
        '''
        db.cursor.execute(query, (date,))
        results = db.cursor.fetchall()

        print(f"Attendance report for {date}:")
        for row in results:
            print(f"Name: {row[0]}, Status: {row[1]}, Time: {row[2]}")
    except Exception as e:
        print(f"Error fetching report: {e}")
    finally:
        db.close()


# Test function
if __name__ == "__main__":
    get_attendance_by_date("2024-06-16")
