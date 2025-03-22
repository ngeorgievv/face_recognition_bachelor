from flask import Flask, Response, render_template
import cv2
import os
import numpy as np  # Import numpy

app = Flask(__name__)

# Initialize OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path where the known faces images are stored
faces_db_path = "faces_db"  # Folder containing images of known people


# Load the face recognizer model
def train_recognizer():
    faces = []
    labels = []
    label_map = {}
    label_count = 0

    for person_name in os.listdir(faces_db_path):
        person_folder = os.path.join(faces_db_path, person_name)
        if os.path.isdir(person_folder):
            # For each person, read their images
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces in the image
                faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces_detected:
                    faces.append(gray[y:y + h, x:x + w])
                    labels.append(label_count)

            # Map the label to the person name
            label_map[label_count] = person_name
            label_count += 1

    # Train the recognizer
    recognizer.train(faces, np.array(labels))
    recognizer.save("face_trainer.yml")
    return label_map


# Train the recognizer at the start
label_map = train_recognizer()

# Open the webcam
camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to grayscale (for face detection)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Crop face region for recognition
                face_region = gray[y:y + h, x:x + w]

                # Recognize the face
                label, confidence = recognizer.predict(face_region)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the name of the recognized person (if confidence is above threshold)
                if confidence < 100:
                    name = label_map[label]
                else:
                    name = "Unknown"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # Display the name
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Convert frame back to BGR for display
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')  # HTML for streaming


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
