import sys
import cv2
import cvzone
import datetime
import os
import pickle
import zlib

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow
from main_ui import Ui_MainWindow  # Import the generated UI file
from counter_mod import Algorithm_Count


class CameraFeedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.stop_btn.setEnabled(False)

        self.file_path = 'Sample Test File\\test_video.mp4'

        self.area1 = [(300, 300), (400, 559), (667, 675), (632, 681)]
        self.area2 = [(110, 400), (313, 566), (579, 703), (624, 694)]

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Connect the "On" button to start the feed
        self.ui.start_btn.clicked.connect(self.start_feed)
        self.ui.stop_btn.clicked.connect(self.stop_feed)

    def start_feed(self):
        self.capture = cv2.VideoCapture(self.file_path)
        # Initialize the video capture and algorithm
        self.algo = Algorithm_Count(self.file_path, self.area1, self.area2, (self.ui.label.width(), self.ui.label.height()))
        self.frame_generator = self.algo.main()  # Initialize the generator
        self.timer.start(10)  # Start the timer to update frames every 30ms
        
        self.ui.start_btn.setEnabled(False)  # Disable the "On" button while the feed is running
        self.ui.stop_btn.setEnabled(True)

    def stop_feed(self):
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
        self.ui.label.setPixmap(QPixmap())
        self.ui.cap_1.setPixmap(QPixmap())
        self.ui.cap_2.setPixmap(QPixmap())
        self.ui.cap_3.setPixmap(QPixmap())
        self.timer.stop()
        self.ui.start_btn.setEnabled(True)
        self.ui.stop_btn.setEnabled(False)

    def update_frame(self):
        try:
            frame, result = next(self.frame_generator)  # Get the next frame from the generator
            self.show_face_crops(frame, self.ui.label)
            self.update_cap()

            self.save_crop_faces()

            

        except StopIteration:
            self.timer.stop()  # Stop the timer when frames are done
            self.ui.start_btn.setEnabled(True)
            self.ui.stop_btn.setEnabled(False)

    def save_crop_faces(self):
        processed_person_ids = set()  # To track already saved person IDs

        try:
            frame, result = next(self.frame_generator)  # Get the next frame from the generator
            
            downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
            directory_name = os.path.join(downloads_path, datetime.datetime.now().strftime('%Y-%m-%d'))
            
            # Ensure the directory exists
            if not os.path.exists(directory_name):
                try:
                    os.makedirs(directory_name)
                    print(f"Directory '{directory_name}' created.")
                except PermissionError:
                    print(f"Permission denied: Unable to create '{directory_name}'.")
                    return
                except Exception as e:
                    print(f"An unexpected error occurred while creating directory '{directory_name}': {e}")
                    return
            
            for person_id, details in result['entering_details'].items():
                # Check if the face crop for this person has already been saved
                if person_id in processed_person_ids:
                    print(f"Skipping already processed Person ID: {person_id}")
                    continue
                
                print(f"Processing Person ID: {person_id}, entered at {details['time']}")
                try:
                    face_crop = pickle.loads(zlib.decompress(details['face_crops']))
                    
                    # Save the face crop to the directory
                    filename = os.path.join(directory_name, f"face_{details['time'].strftime('%H-%M-%S')}.jpg")
                    cv2.imwrite(filename, face_crop)
                    print(f"Saved face crop to {filename}")
                    
                    # Mark this person_id as processed
                    processed_person_ids.add(person_id)
                except Exception as e:
                    print(f"An error occurred while processing Person ID {person_id}: {e}")
        
        except StopIteration:
            print("No more frames to process.")


    def show_face_crops(self, face_crops, name_label):
        face_resized = cv2.resize(face_crops, (name_label.width(), name_label.height()), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_height, face_width, face_channels = face_rgb.shape
        face_bytes_per_line = face_channels * face_width
        face_qimg = QImage(face_rgb.data, face_width, face_height, face_bytes_per_line, QImage.Format_RGB888)
        face_pixmap = QPixmap.fromImage(face_qimg)
        name_label.setPixmap(face_pixmap)

    def update_cap(self):
        """
        Update the camera feed with the latest frame and display face crops.
        This method retrieves the next frame from the frame generator and processes
        the face crops detected in the frame. It displays up to three face crops in
        the UI.
        Raises:
            StopIteration: If the frame generator has no more frames to provide.
        How it works:
        1. Retrieves the next frame and result from the frame generator.
        2. Initializes an empty list to store face crops.
        3. Iterates over the detected persons and their details, printing the person ID and entry time.
        4. Inserts the face crops into the list.
        5. If there are face crops, decompresses and displays up to three face crops in the UI.
        """
        try:
            frame, result = next(self.frame_generator)  # Get the next frame from the generator

            # print(result)
            temp = []
        
            # Access face crops from the result
            for person_id, details in result['entering_details'].items():
                print(f"Person ID: {person_id}, entered at {details['time']}")
                temp.insert(0, details['face_crops'])

            if temp:
                x = temp[0] # Get the face crop
                x1 = pickle.loads(zlib.decompress(x)) # Decompress the face crop
                self.show_face_crops(x1, self.ui.cap_1) # Display the face crop in the QLabel
                if len(temp) > 1:
                    y = temp[1]
                    y1 = pickle.loads(zlib.decompress(y))
                    self.show_face_crops(y1, self.ui.cap_2)
                if len(temp) > 2:
                    z = temp[2]
                    z1 = pickle.loads(zlib.decompress(z))
                    self.show_face_crops(z1, self.ui.cap_3)
        except StopIteration:
            pass

    def closeEvent(self, event):
        """Release the video capture when the window is closed."""
        self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraFeedWindow()
    window.show()
    sys.exit(app.exec())