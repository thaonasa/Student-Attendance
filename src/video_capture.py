
import cv2
from face_detection import detect_faces
from face_recognition import encode_face, recognize_face
from connect_database import DatabaseConnection
from datetime import datetime
from config import VIDEO_SOURCE, CASCADE_PATH


def start_video_capture(video_source=VIDEO_SOURCE):
    """
    Start video capture from the given video source.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return None
    print("Video capture started.")
    return cap


def capture_frame(cap):
    """
    Capture a frame from the webcam feed.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        return None
    return frame


def stop_video_capture(cap):
    """
    Stop the video capture.
    """
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("Video capture stopped.")


def mark_attendance(db, student_id):
    """
    Mark attendance for the given student in the database.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "Present"

        db.cursor.execute(
            "INSERT INTO Attendance (student_id, status, created_at) VALUES (?, ?, ?)",
            (student_id, status, timestamp)
        )
        db.connection.commit()
        print(f"Attendance marked for Student ID: {student_id}")
    except Exception as e:
        print(f"Error marking attendance: {e}")


def process_video():
    """
    Process the video feed to detect and recognize faces, then mark attendance.
    """
    # Connect to the database
    db = DatabaseConnection()
    db.connect()

    # Start video capture
    cap = start_video_capture()
    if not cap:
        return

    try:
        while True:
            # Capture a frame
            frame = capture_frame(cap)
            if frame is None:
                break

            # Detect faces in the frame
            faces = detect_faces(frame, CASCADE_PATH)

            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract the face region
                face_region = frame[y:y+h, x:x+w]

                # Encode and recognize the face
                face_encoding = encode_face(face_region)
                student_id = recognize_face(face_encoding)

                if student_id:
                    mark_attendance(db, student_id)
                    # Display the student ID on the video feed
                    cv2.putText(frame, f"ID: {student_id}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show the video feed with detections
            cv2.imshow("Video Feed", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        # Stop video capture and close database connection
        stop_video_capture(cap)
        db.close()


if __name__ == "__main__":
    process_video()
