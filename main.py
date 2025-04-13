import face_recognition
import csv
import numpy as np
import cv2
from datetime import datetime

# Start webcam
video_capture = cv2.VideoCapture(0)

# Load known faces
manas_image = face_recognition.load_image_file("faces/manas.jpg")
manas_encoding = face_recognition.face_encodings(manas_image)[0]

known_face_encodings = [manas_encoding]
known_face_names = ["Manas"]

# List of expected students
students = known_face_names.copy()

# For storing detected face info
face_locations = []
face_encodings = []

# Get current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create CSV file for the day
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distance)
        name = "Unknown"

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Display name on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,
                        f"{name} Present",
                        (10, 100),
                        font, 1.5,
                        (255, 0, 0),
                        3,
                        cv2.LINE_AA)

            # Mark attendance
            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    # Show the frame
    cv2.imshow("Attendance", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
f.close()
