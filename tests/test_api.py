import requests
import os

# URL của API
BASE_URL = "http://127.0.0.1:8000"

# Các đường dẫn test
TEST_IMAGE_PATH = '/data/Original Images/Test/Akshay Kumar/gettyimages-107449876-612x612_face1.jpg'
INVALID_IMAGE_PATH = 'D:/Cá nhân/MSE/Xử lý ảnh và video/Final_Project/data/Invalid Images/invalid_image.txt'
EMPTY_IMAGE_PATH = 'D:\Cá nhân\MSE\Xử lý ảnh và video\Final_Project\data\Empty Images\empty_image.jpg'


def test_root_endpoint():
    """Kiểm tra endpoint root."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to the Face Recognition API!"
    print("Test Root Endpoint: Passed")


def test_recognize_face_valid():
    """Kiểm tra API nhận diện khuôn mặt với ảnh hợp lệ."""
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image does not exist: {TEST_IMAGE_PATH}")
        return

    with open(TEST_IMAGE_PATH, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/recognize/",
            files={"file": file}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_class" in data
        assert "confidence" in data
        print("Test Recognize Face Valid:", data)


def test_recognize_face_invalid():
    """Kiểm tra API nhận diện khuôn mặt với ảnh không hợp lệ."""
    if not os.path.exists(INVALID_IMAGE_PATH):
        print(f"Invalid image does not exist: {INVALID_IMAGE_PATH}")
        return

    with open(INVALID_IMAGE_PATH, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/recognize/",
            files={"file": file}
        )
        assert response.status_code == 400
        assert "Invalid image format" in response.json()["detail"]
        print("Test Recognize Face Invalid: Passed")


def test_detect_faces_valid():
    """Kiểm tra API phát hiện khuôn mặt với ảnh hợp lệ."""
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image does not exist: {TEST_IMAGE_PATH}")
        return

    with open(TEST_IMAGE_PATH, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/detect_faces/",
            files={"file": file}
        )
        assert response.status_code == 200
        data = response.json()
        assert "faces" in data
        print("Test Detect Faces Valid:", data)


def test_detect_faces_no_faces():
    """Kiểm tra API phát hiện khuôn mặt với ảnh không chứa khuôn mặt."""
    if not os.path.exists(EMPTY_IMAGE_PATH):
        print(f"Empty image does not exist: {EMPTY_IMAGE_PATH}")
        return

    with open(EMPTY_IMAGE_PATH, "rb") as file:
        response = requests.post(
            f"{BASE_URL}/detect_faces/",
            files={"file": file}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "No faces detected."
        print("Test Detect Faces No Faces: Passed")


if __name__ == "__main__":
    # Kiểm tra root endpoint
    test_root_endpoint()

    # Kiểm tra endpoint nhận diện khuôn mặt
    test_recognize_face_valid()
    test_recognize_face_invalid()

    # Kiểm tra endpoint phát hiện khuôn mặt
    test_detect_faces_valid()
    test_detect_faces_no_faces()
