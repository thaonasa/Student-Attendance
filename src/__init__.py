# __init__.py for src package

# Import các module chính để tiện sử dụng
from .config import CASCADE_PATH
from .face_detection import detect_faces
from .video_capture import capture_video

# Xuất các module khi package được import
__all__ = ["config", "face_detection", "video_capture"]

# Thông báo khi package được import (tuỳ chọn)
print("The 'src' package has been loaded successfully!")

# Có thể cấu hình global constants (nếu cần)
PACKAGE_NAME = "src"
VERSION = "1.0.0"
