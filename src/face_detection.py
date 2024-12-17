from config import CASCADE_PATH
from connect_database.connect_db import DatabaseConnection
import cv2
import sys
import os

# # Thêm đường dẫn gốc của dự án vào sys.path
# project_root = os.path.abspath(os.path.join(
#     os.path.dirname(__file__), "..", ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Tránh vòng lặp, đảm bảo chỉ import những gì cần thiết


def load_haar_cascade():
    """
    Load the Haar cascade classifier.
    :return: Preloaded Haar cascade classifier.
    """
    if not os.path.isfile(CASCADE_PATH):
        raise FileNotFoundError(
            f"Error: Haar cascade file not found at {CASCADE_PATH}")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise Exception(
            f"Error: Unable to load Haar cascade from {CASCADE_PATH}")

    print(f"Haar cascade loaded successfully from: {CASCADE_PATH}")
    return face_cascade


def detect_faces(image, face_cascade):
    """
    Detect faces in an image using Haar cascade.
    :param image: Input image (numpy array).
    :param face_cascade: Preloaded Haar cascade classifier.
    :return: List of bounding boxes for detected faces [(x, y, w, h), ...].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces


def detect_and_crop_faces_haar(input_dir, output_dir):
    """
    Detect and crop faces from all images in the input directory using Haar cascades,
    and save them in the output directory while maintaining the folder structure.

    :param input_dir: Path to the directory containing image subdirectories.
    :param output_dir: Path to save cropped face images while maintaining the structure.
    """
    face_cascade = load_haar_cascade()

    # Walk through the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)

                # Generate corresponding output path
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Read the image
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Error reading image: {input_path}")
                    continue

                # Detect faces
                faces = detect_faces(image, face_cascade)

                # Process and save cropped faces
                if len(faces) > 0:
                    for i, (x, y, w, h) in enumerate(faces):
                        cropped_face = image[y:y+h, x:x+w]
                        face_output_path = os.path.join(
                            os.path.dirname(output_path),
                            f"{os.path.splitext(file)[0]}_face{i}.jpg"
                        )
                        cv2.imwrite(face_output_path, cropped_face)
                        print(f"Saved cropped face: {face_output_path}")
                else:
                    print(f"No faces detected in: {input_path}")


if __name__ == "__main__":
    try:
        print("Starting face detection and cropping...")
        detect_and_crop_faces_haar(
            input_dir="data/Original Images",  # Update input directory
            output_dir="data/Output"          # Update output directory
        )
        print("Face detection completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
