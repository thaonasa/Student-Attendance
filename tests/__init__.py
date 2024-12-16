# __init__.py for tests package

# Import các module hoặc file test chính
from .test_api import test_root_endpoint, test_recognize_face_valid, test_detect_faces_valid

# Xuất các thành phần khi gọi from tests import *
__all__ = ["test_root_endpoint",
           "test_recognize_face_valid", "test_detect_faces_valid"]

# Thông báo khi package được load (tuỳ chọn)
print("The 'tests' package has been loaded successfully!")

# Thông tin package
PACKAGE_NAME = "tests"
VERSION = "1.0.0"
