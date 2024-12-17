from connect_db import DatabaseConnection

def init_database():
    """Khởi tạo cơ sở dữ liệu và tạo các bảng cần thiết."""
    db = DatabaseConnection()
    db.connect()
    db.create_tables()
    db.close()

if __name__ == "__main__":
    init_database()
    print("Database initialized successfully!") 