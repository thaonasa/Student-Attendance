import os
import cv2
from config import CASCADE_PATH


def detect_and_crop_faces_haar(input_dir, output_dir):
    """
    Detect and crop faces from all images in the input directory using Haar cascades,
    and save them in the output directory while maintaining the folder structure.

    :param input_dir: Path to the directory containing 'Train', 'Validation', and 'Test' subdirectories.
    :param output_dir: Path to save cropped face images while maintaining the structure of 'Train', 'Validation', and 'Test'.
    """
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if not face_cascade:
        print(f"Error loading Haar cascade from: {CASCADE_PATH}")
        return

    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
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

                # Convert to grayscale for Haar cascade
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                # Process detections
                face_detected = False
                for i, (x, y, w, h) in enumerate(faces):
                    cropped_face = image[y:y+h, x:x+w]

                    # Save the cropped face
                    face_output_path = os.path.join(os.path.dirname(
                        output_path), f"{os.path.splitext(file)[0]}_face{i}.jpg")
                    cv2.imwrite(face_output_path, cropped_face)
                    face_detected = True
                    print(f"Saved cropped face: {face_output_path}")

                if not face_detected:
                    print(f"No faces detected in: {input_path}")


if __name__ == "__main__":
    # Update the input and output directories as needed
    detect_and_crop_faces_haar(
        # Root directory containing Train, Validation, and Test
        input_dir="data/Original Images",
        output_dir="data/Output"          # Directory to save cropped faces
    )
