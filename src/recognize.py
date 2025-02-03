import sys
import numpy as np
import torch
from torchvision import transforms
from freenect import sync_get_video, sync_get_depth
import json
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QInputDialog, QPushButton, QDialog, QVBoxLayout, QHBoxLayout  # Added additional dialogs and layout imports
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import cv2
from model import FaceShapeRecognitionModel  # Import the FaceShapeRecognitionModel
import face_recognition  # Import face_recognition library
import os  # Import os for setting environment variable

# Configuration
MODEL_PATH = "model.pth"
IMAGE_SIZE = (256, 256)
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to recognize a face
DEPTH_VARIANCE_THRESHOLD = 100.0  # Minimum variance in depth values to consider valid

# Load the trained model
def load_model():
    label_mapping = {0: "negative", 1: "positive"}
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # Add reverse mapping

    num_classes = len(label_mapping)
    model = FaceShapeRecognitionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()  # Set model to evaluation mode
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, label_mapping, reverse_label_mapping  # Return reverse mapping

# Preprocessing function for live feed
def preprocess_frame(rgb_frame, depth_frame):
    depth_frame_normalized = (depth_frame / depth_frame.max() * 255).astype(np.uint8)

    transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    transform_depth = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    rgb_tensor = transform_rgb(rgb_frame).unsqueeze(0)  # Add batch dimension
    depth_tensor = transform_depth(depth_frame_normalized).unsqueeze(0)  # Add batch dimension

    return rgb_tensor, depth_tensor

# Depth consistency check
def is_valid_depth(depth_frame):
    depth_variance = np.var(depth_frame)
    return depth_variance >= DEPTH_VARIANCE_THRESHOLD

# Predict function with confidence threshold
def predict(model, depth_tensor, label_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_tensor = depth_tensor.to(device)

    with torch.no_grad():
        outputs = model(depth_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        max_confidence, predicted_class = torch.max(probabilities, dim=1)

    if max_confidence.item() >= CONFIDENCE_THRESHOLD:
        predicted_label = label_mapping.get(predicted_class.item(), "unknown")
        return predicted_label, max_confidence.item(), probabilities.cpu().numpy()
    else:
        return None, None, probabilities.cpu().numpy()

# PyQt5 Application
class KinectApp(QMainWindow):
    def __init__(self, model, label_mapping, reverse_label_mapping):
        super().__init__()
        self.model = model
        self.label_mapping = label_mapping
        self.reverse_label_mapping = reverse_label_mapping  # Store reverse mapping

        # Set up the main window
        self.setWindowTitle("Kinect RGB Feed")
        self.setStyleSheet("background-color: black;")  # Set background to black

        # Create a QLabel to display the video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.video_label)  # Ensure QLabel is the central widget

        # Set up a QTimer to update the video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (~33 FPS)

        # Remove toolbar button and add a bottom right scan button
        self.scan_button = QPushButton("Register/Update", self)
        self.scan_button.clicked.connect(self.start_scan)
        self.scan_button.setStyleSheet("color: white; padding: 5px;")  # Changed: text color white

    def resizeEvent(self, event):
        """Resize QLabel to fill the entire window."""
        self.video_label.setGeometry(self.rect())
        # Position scan button at bottom right with 10px margin
        btn_size = self.scan_button.sizeHint()
        x = self.width() - btn_size.width() - 10
        y = self.height() - btn_size.height() - 10
        self.scan_button.move(x, y)

    def update_frame(self):
        # Get RGB and Depth frames
        rgb_frame, _ = sync_get_video()
        depth_frame, _ = sync_get_depth()

        if rgb_frame is None or depth_frame is None:
            print("Error: Could not get frames from Kinect.")
            return

        rgb_frame = np.array(rgb_frame)
        depth_frame = np.array(depth_frame)

        # Detect face before making predictions
        haarcascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(haarcascade_path)
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        people_count = 0
        recognized_people = []

        for (x, y, w, h) in faces:
            face_rgb = rgb_frame[y:y + h, x:x + w]
            face_depth = depth_frame[y:y + h, x:x + w]

            # Check depth consistency
            if is_valid_depth(face_depth):
                people_count += 1
                # Preprocess the frames and perform depth-based prediction (model only determines if face is real or flat)
                _, depth_tensor = preprocess_frame(face_rgb, face_depth)
                predicted_label, confidence, probabilities = predict(self.model, depth_tensor, self.label_mapping)
                recognized_people.append(predicted_label if predicted_label else "unknown")
                
                # Use face_recognition for 2D name detection
                face_locations = face_recognition.face_locations(face_rgb)
                if face_locations:
                    face_rgb_converted = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
                    face_encodings = face_recognition.face_encodings(face_rgb_converted)
                    name = "unknown"
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        if True in matches:
                            name = known_face_names[matches.index(True)]
                    
                    # Overlay text based on model prediction and recognized name
                    if name != "unknown":
                        if predicted_label == "positive":
                            cv2.putText(rgb_frame, f"3d: {name}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        elif predicted_label == "negative":
                            cv2.putText(rgb_frame, f"flat image: {name}", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        if predicted_label == "positive":
                            cv2.putText(rgb_frame, "3d: positive", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        # If predicted_label is negative and no name, do not overlay any text.
                else:
                    if predicted_label == "positive":
                        cv2.putText(rgb_frame, "3d: positive", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # If predicted_label is negative, do not overlay any text.

        # Display the number of people detected and their names
        message = f"I see {people_count} person(s)"
        cv2.putText(rgb_frame, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Add depth preview overlay in the bottom left corner
        depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        preview_w, preview_h = 200, 150  # preview width and height
        depth_preview = cv2.resize(depth_colored, (preview_w, preview_h))
        y_offset = rgb_frame.shape[0] - preview_h - 10  # 10px margin from bottom
        x_offset = 10  # 10px margin from left
        rgb_frame[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = depth_preview

        # Convert frame to QImage for PyQt5 display
        rgb_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Update QLabel with the scaled image
        self.video_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    # New method: open a dialog to get user name and start scanning.
    def start_scan(self):
        # Pause the live feed to free the camera
        self.timer.stop()
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Scan User")
        dialog.setLabelText("Enter user name:")
        # Set style for QLineEdit, QLabel, and QPushButton within the dialog.
        dialog.setStyleSheet(
            "QLineEdit { color: white; background-color: black; } "
            "QLabel { color: white; } "
            "QPushButton { color: white; background-color: gray; }"
        )
        if dialog.exec_() == QDialog.Accepted:
            name = dialog.textValue().strip()
            if name:
                reg_window = RegistrationWindow(name)
                reg_window.exec_()
        # Resume the live feed after registration is done
        self.timer.start(30)

# New class for registration scanning as its own UI
class RegistrationWindow(QDialog):
    def __init__(self, user_name):
        super().__init__()
        self.user_name = user_name
        self.setWindowTitle(f"Register/Update - {user_name}")
        self.setStyleSheet("background-color: black; color: white;")
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        # Move info label below preview_label with fixed size
        self.info_label = QLabel("Ready", self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFixedHeight(30)  # limited height
        self.capture_button = QPushButton("Capture Burst", self)
        self.quit_button = QPushButton("Quit", self)
        self.capture_button.setStyleSheet("color: white; background-color: gray; padding: 5px;")
        self.quit_button.setStyleSheet("color: white; background-color: gray; padding: 5px;")
        # Layout: preview on top, then control row (info + buttons)
        layout = QVBoxLayout()
        layout.addWidget(self.preview_label)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.info_label)
        control_layout.addWidget(self.capture_button)
        control_layout.addWidget(self.quit_button)
        layout.addLayout(control_layout)
        self.setLayout(layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        # Burst capture logic remains the same
        self.capture_button.clicked.connect(self.start_burst_capture)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.count = 0
        self.burst_count = 0
        self.total_burst = 5
        self.burst_timer = QTimer(self)
        self.burst_timer.timeout.connect(self.burst_capture)
        self.countdown = 3
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.quit_button.clicked.connect(self.close)
    
    # New method: start burst capture with countdown
    def start_burst_capture(self):
        self.capture_button.setEnabled(False)
        self.countdown = 3
        self.info_label.setText(f"Burst capture starting in {self.countdown}...")
        self.countdown_timer.start(1000)  # update every second
    
    def update_countdown(self):
        self.countdown -= 1
        if self.countdown > 0:
            self.info_label.setText(f"Burst capture starting in {self.countdown}...")
        else:
            self.countdown_timer.stop()
            self.info_label.setText("Capturing burst...")
            self.burst_count = 0
            self.burst_timer.start(500)  # capture every 500ms
    
    def burst_capture(self):
        rgb_frame, _ = sync_get_video()
        if rgb_frame is None:
            print("Burst capture failed: no frame retrieved.")
            return
        rgb_frame = np.array(rgb_frame)
        h, w, _ = rgb_frame.shape
        # Crop center 256x256 region
        cx, cy = w // 2, h // 2
        half_size = 128
        cropped = rgb_frame[max(0, cy - half_size):cy + half_size, max(0, cx - half_size):cx + half_size]
        save_folder = f"/home/rogelio/FRS/3d-2d-face-recognition/saved_people/{self.user_name}"
        os.makedirs(save_folder, exist_ok=True)
        img_path = os.path.join(save_folder, f"{self.user_name}_burst_{self.burst_count}.jpg")
        cv2.imwrite(img_path, cropped)
        print(f"Burst captured image {self.burst_count}")
        self.burst_count += 1
        if self.burst_count >= self.total_burst:
            self.burst_timer.stop()
            self.info_label.setText("Burst capture complete!")
            self.capture_button.setEnabled(True)
            # Optionally reset info message after a delay
            QTimer.singleShot(2000, lambda: self.info_label.setText("Ready"))
    
    def update_frame(self):
        # Use freenect to get the video frame from Kinect
        rgb_frame, _ = sync_get_video()
        if rgb_frame is None:
            return
        rgb_frame = np.array(rgb_frame)
        # Draw a guide rectangle at the center (256x256)
        h, w, _ = rgb_frame.shape
        cx, cy = w // 2, h // 2
        top_left = (max(0, cx - 128), max(0, cy - 128))
        bottom_right = (min(w, cx + 128), min(h, cy + 128))
        cv2.rectangle(rgb_frame, top_left, bottom_right, (0, 255, 0), 2)
        # Convert for display
        rgb_display = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        h_img, w_img, ch = rgb_display.shape
        bytes_per_line = ch * w_img
        q_image = QImage(rgb_display.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
        self.preview_label.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio))
    
    def closeEvent(self, event):
        if self.timer is not None:
            self.timer.stop()
        event.accept()

def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    known_face_encodings.append(face_encodings[0])
                    # Use the name of the subfolder as the person's name
                    person_name = os.path.basename(root)
                    known_face_names.append(person_name)

    return known_face_encodings, known_face_names

# New scan_users now accepts an optional user_name parameter.
def scan_users(user_name=None):
    if not user_name:
        user_name = input("Enter the user name: ").strip()
    save_folder = f"/home/rogelio/FRS/3d-2d-face-recognition/saved_people/{user_name}"
    os.makedirs(save_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Press 'c' to capture, 'q' to quit.")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Scan - Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            if len(faces) > 0:
                img_path = os.path.join(save_folder, f"{user_name}_{count}.jpg")
                cv2.imwrite(img_path, frame)
                print(f"Captured image {count}")
                count += 1
            else:
                print("No face detected. Try again.")
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished capturing {count} image(s) for user: {user_name}")

# Main function
def main():
    # Set the environment variable for Qt platform plugin
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/rogelio/.local/lib/python3.10/site-packages/cv2/qt/plugins/platforms"

    model, label_mapping, reverse_label_mapping = load_model()

    # Load known faces and their encodings
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces("/home/rogelio/FRS/3d-2d-face-recognition/saved_people")

    app = QApplication(sys.argv)
    window = KinectApp(model, label_mapping, reverse_label_mapping)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
