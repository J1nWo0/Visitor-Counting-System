import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
import torch
import datetime  # Import for timestamp

# Define a class to manage colors used in the application
class Color:
    def __init__(self):
        # Define a dictionary to store various colors used in the application
        self.colors = {
            'boundingBox1': (0, 255, 0),       # Green color for bounding box 1
            'boundingBox2': (0, 255, 255),     # Yellow color for bounding box 2
            'text1': (255, 255, 255),          # White color for primary text
            'text2': (0, 0, 0),                # Black color for secondary text
            'area1': (255, 0, 0),              # Blue color for area 1
            'area2': (0, 0, 255),              # Red color for area 2
            'point': (255, 0, 255),            # Magenta color for points
            'center_point': (255, 255, 0),     # Cyan color for center points
            'rectangle': (0, 119, 255),        # Orange color for rectangles
            'mask': (128, 128, 128)            # Gray color for masks
        }

    def __getattr__(self, item):
        return lambda: self.colors[item]
# Initialize color manager
color = Color()

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
# Load YOLO model and move it to the appropriate device
model_person = YOLO('yolo-Weights/yolo11n.pt').to(device)  # Model for person segmentation
model_face = YOLO('yolo-Weights/yolov10n-face.pt').to(device)   # Model for face detection


# Define a class to handle the counting algorithm
class Algorithm_Count:
    def __init__(self, file_path, a1, a2, frame_size):
        """
        Initializes the counter object with the given parameters.
        Args:
            file_path (str): The path to the file where data will be read.
            a1 (tuple): Coordinates defining the first area of interest.
            a2 (tuple): Coordinates defining the second area of interest.
            frame_size (tuple): The size of the video frame.
        Attributes:
            peopleEntering (dict): Dictionary to keep track of people entering.
            entering (set): Set to keep track of entering people.
            peopleExiting (dict): Dictionary to keep track of people exiting.
            exiting (set): Set to keep track of exiting people.
            file_path (str): Path to the file where data will be read.
            area1 (tuple): Coordinates defining the first area of interest.
            area2 (tuple): Coordinates defining the second area of interest.
            frame_size (tuple): The size of the video frame.
            paused (bool): Flag to indicate if the system is paused.
            coordinates (list): List to store coordinates.
            name_frame (str): Name of the window displaying frames.
            start_time (float): Start time of the system.
        """
        self.peopleEntering = {}    
        self.entering = set()
        self.peopleExiting = {}
        self.exiting = set()
        self.file_path = file_path
        self.area1 = a1
        self.area2 = a2
        self.frame_size = frame_size
        self.paused = False
        self.coordinates = []
        self.name_frame = 'People Counting System'
        self.start_time = time.time()

        # Create a named window for displaying frames
        cv2.namedWindow(self.name_frame)

    # Method to detect objects in a frame
    def detect_BboxOnly(self, frame):
        """
        Detects persons and faces in a given video frame using pre-trained models.
        Args:
            frame (numpy.ndarray): The input video frame in which to detect persons and faces.
        Returns:
            tuple: A tuple containing two lists:
                - person_detections (list): A list of detected persons with their bounding boxes.
                - face_detections (list): A list of detected faces with their bounding boxes.
        # The function uses two different models to detect persons and faces in the input frame.
        # It processes the detection results and returns the bounding boxes for both persons and faces.
        """
        # Detect persons and faces using different models
        results_person = model_person.track(frame, conf=0.6, classes=[0], persist=True, tracker="bytetrack.yaml")  # Detect persons only (class 0)
        results_face = model_face.track(frame, conf=0.6, classes=[0], persist=True, tracker="bytetrack.yaml")  # Detect faces

        # Process results
        person_detections = self.process_results(results_person)
        face_detections = self.process_results(results_face)

        return person_detections, face_detections

    # Method to process the detection results
    def process_results(self, results):
        """
        Processes the detection results and extracts relevant information.
        Args:
            results (list): A list of detection results, where each result contains bounding boxes and segmentation masks.
        Returns:
            list: A list of detections, where each detection is represented as a list containing:
                - x1 (int): Top-left x-coordinate of the bounding box.
                - y1 (int): Top-left y-coordinate of the bounding box.
                - x2 (int): Bottom-right x-coordinate of the bounding box.
                - y2 (int): Bottom-right y-coordinate of the bounding box.
                - box_id (int): Tracking ID of the bounding box (or -1 if not assigned).
                - class_id (int): Class index of the detected object (or -1 if not assigned).
                - score (float): Confidence score of the detection (or 0.0 if not available).
                - mask (numpy.ndarray or None): Segmentation mask as a numpy array (or None if not available).
        The function iterates through the detection results, extracts bounding box coordinates, tracking IDs, confidence scores,
        class indices, and segmentation masks (if available), and appends them to the detections list.
        """
        detections = []
        for r in results:  # Iterate through the results
            boxes = r.boxes  # Extract bounding boxes
            masks = r.masks  # Extract segmentation masks (if present)
            for mask_id, box in enumerate(boxes):
                # Extracting bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Get the top-left (x1, y1) and bottom-right (x2, y2) coordinates
                # Extracting the tracking ID (if present)
                box_id = box.id if box.id is not None else -1  # ID will be -1 if there's no ID assigned
                # Extracting confidence score
                score = box.conf[0] if box.conf is not None else 0.0  # Default to 0.0 if confidence is missing
                # Extracting the class index (if present)
                class_id = box.cls[0] if box.cls is not None else -1  # Default to -1 if class is missing
                # Extract segmentation mask if available (convert mask to numpy array)
                mask = np.array(masks[mask_id].xy, dtype=np.int32) if masks is not None else None  # Convert mask to numpy array if it exists
                # Append to the detections list as a tuple
                detections.append([int(x1), int(y1), int(x2), int(y2), int(box_id), int(class_id), float(score), mask])

        return detections
    
    # Method to display elapsed time on the frame
    def show_time(self, frame):
        elapsed_time = time.time() - self.start_time

        # Convert elapsed time to hours, minutes, seconds, and milliseconds
        milliseconds = int(elapsed_time * 1000)
        hours, milliseconds = divmod(milliseconds, 3600000)
        minutes, milliseconds = divmod(milliseconds, 60000)
        seconds = (milliseconds / 1000.0)

        # Display the time in the format "hour:minute:second.millisecond"
        time_str = "Running Time: {:02}:{:02}:{:06.3f}".format(int(hours), int(minutes), seconds)
        cvzone.putTextRect(frame, time_str, (20, 480), 1, 1, color.text1(), color.text2())

    def change_coord_point(self, x1, x2, y1, y2):
        """
        Adjusts the coordinates of a point based on the given parameters.

        This function calculates a new point that is positioned at 50% of the 
        horizontal distance from the left edge (x1) to the right edge (x2), 
        and 4% of the vertical distance from the bottom edge (y2) to the top edge (y1).

        Args:
            x1 (int): The x-coordinate of the left edge.
            x2 (int): The x-coordinate of the right edge.
            y1 (int): The y-coordinate of the top edge.
            y2 (int): The y-coordinate of the bottom edge.

        Returns:
            tuple: A tuple containing the new x and y coordinates (new_x, new_y).
        """
        new_x = int(x1 + (x2 - x1) * 0.5)  # 50% from the left edge
        new_y = int(y2 - (y2 - y1) * 0.04)  # 4% from the bottom edge
        return new_x, new_y

    # Method to count people entering and exiting
    def counter(self, frame, detections_person, detections_face):
        """
        Processes the given frame to count and track people entering and exiting.
        Args:
            frame (numpy.ndarray): The current video frame.
            detections_person (list): A list of detections for persons, where each detection is a tuple containing 
                                      (x1, y1, x2, y2, box_id, score, class_id, mask).
            detections_face (list): A list of detections for faces.
        This function performs the following steps:
        1. Iterates over each person detection.
        2. Extracts bounding box coordinates, box ID, score, class ID, and mask from the detection.
        3. Creates a label for the detected person.
        4. Draws bounding boxes for persons and faces on the frame.
        5. Tracks people entering and exiting based on the bounding box coordinates and box ID.
        6. Draws polylines on the frame for visualization.
        Returns:
            None
        """
        for detect in detections_person:
            x1, y1, x2, y2, box_id, class_id, score, mask = detect
            label = f"{box_id} Person: {score:.2f}"
            
            self.person_bounding_boxes(frame, x1, y1, x2, y2, box_id, class_id, score, mask)
            self.face_bounding_boxes(frame, detections_face)
            self.track_people_entering(frame, x1, y1, x2, y2, box_id, label)
            self.track_people_exiting(frame, x1, y1, x2, y2, box_id, label)
            
        self.draw_polylines(frame)

    # Method to draw bounding boxes around detected persons
    def person_bounding_boxes(self, frame, x1, y1, x2, y2, box_id, class_id, score, mask):
        """
        Draws bounding boxes, text, and masks on a given frame for detected persons.

        Parameters:
        frame (numpy.ndarray): The image frame on which to draw.
        x1 (int): The x-coordinate of the top-left corner of the bounding box.
        y1 (int): The y-coordinate of the top-left corner of the bounding box.
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
        box_id (int): The ID of the bounding box. If -1, the box is not drawn.
        class_id (int): The class ID of the detected object.
        score (float): The confidence score of the detection.
        mask (numpy.ndarray or None): The mask for the detected object. If None, no mask is drawn.

        Returns:
        None

        """
        # Draws bounding boxes, text, and masks on a given frame for detected persons.
        if box_id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.rectangle(), 2)
            cvzone.putTextRect(frame, f"{class_id}: {box_id}: {score:.2f}", (x1, y1 - 10), 1, 1, color.text1(), color.text2())
            cx, cy = self.change_coord_point(x1, x2, y1, y2)
            cv2.circle(frame, (cx, cy), 4, color.point(), -1)  
            # Check if mask is valid and draw it
            if mask is not None:
                # cv2.fillPoly(frame, [mask], color.mask()) # Fill the mask with a color
                cv2.polylines(frame, [mask], True, color.center_point(), 2)  # Draw the mask outline

    def face_bounding_boxes(self, frame, face_detections):
        for detect in face_detections:
            x1, y1, x2, y2, box_id, class_id, score, mask = detect
            label = f"{box_id} Face: {score:.2f}"
            if box_id != -1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.center_point(), 1)
                # cvzone.putTextRect(frame, f"{class_id}: {box_id}: {score:.2f}", (x1, y1 - 10), 1, 1, color.text1(), color.text2())
                # Check if mask is valid and draw it
                if mask is not None:
                    # cv2.fillPoly(frame, [mask], color.mask()) # Fill the mask with a color
                    cv2.polylines(frame, [mask], True, color.center_point(), 1)  # Draw the mask outline

    # Method to track people entering a specified area
    def track_people_entering(self, frame, x1, y1, x2, y2, id, label):
        """
        Tracks people entering a specified area in a video frame.

        This function checks if a person (identified by a bounding box and an ID) 
        has entered a specified area (self.area2) and records the entry time. 
        If the person is already recorded as entering, it further checks if the 
        person has moved into another specified area (self.area1) and updates 
        the frame with visual indicators.

        Args:
            frame (numpy.ndarray): The current video frame.
            x1 (int): The x-coordinate of the top-left corner of the bounding box.
            y1 (int): The y-coordinate of the top-left corner of the bounding box.
            x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
            y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
            id (int): The unique identifier for the person being tracked.
            label (str): The label to be displayed on the bounding box.

        Returns:
            None
        """
        cx, cy = self.change_coord_point(x1, x2, y1, y2)
        result_p1 = cv2.pointPolygonTest(np.array(self.area2, np.int32), ((cx, cy)), False)
        if result_p1 >= 0:
            self.peopleEntering[id] = {'coords': (cx, cy), 'time': datetime.datetime.now()}  # Add timestamp
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox2(), 2)
            cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2()) 
        if id in self.peopleEntering:
            result_p2 = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((cx, cy)), False)
            if result_p2 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox1(), 2)
                cv2.circle(frame, (cx, cy), 4, color.point(), -1)  
                cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2())
                self.entering.add((id, self.peopleEntering[id]['time']))

    # Method to track people exiting a specified area
    def track_people_exiting(self, frame, x1, y1, x2, y2, id, label):
        """
        Tracks people exiting a defined area in a video frame.

        Args:
            frame (numpy.ndarray): The current video frame.
            x1 (int): The x-coordinate of the top-left corner of the bounding box.
            y1 (int): The y-coordinate of the top-left corner of the bounding box.
            x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
            y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
            id (int): The unique identifier for the person being tracked.
            label (str): The label to display on the bounding box.

        Returns:
            None

        This function checks if a person is within a defined exit area (self.area1) and updates their coordinates and timestamp.
        If the person is within another defined area (self.area2), it draws a bounding box and a circle on the frame, 
        and adds the person's ID and timestamp to the exiting set.
        """
        cx, cy = self.change_coord_point(x1, x2, y1, y2)
        result_p3 = cv2.pointPolygonTest(np.array(self.area1, np.int32), ((cx, cy)), False)
        if result_p3 >= 0:
            self.peopleExiting[id] = {'coords': (cx, cy), 'time': datetime.datetime.now()}  # Add timestamp
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox1(), 2)
            cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2()) 
        if id in self.peopleExiting:
            result_p4 = cv2.pointPolygonTest(np.array(self.area2, np.int32), ((cx, cy)), False)
            if result_p4 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox2(), 2)
                cv2.circle(frame, (cx, cy), 4, color.point(), -1)  
                cvzone.putTextRect(frame, label, (x1 + 10, y1 - 10), 1, 1, color.text1(), color.text2())
                self.exiting.add((id, self.peopleExiting[id]['time']))

    # Method to draw polylines for specified areas and display counts
    def draw_polylines(self, frame):
        """
        Draws polylines on the given frame to represent specified areas and overlays text indicating the number of entries and exits.

        Args:
            frame (numpy.ndarray): The image frame on which to draw the polylines and text.

        This function performs the following steps:
        1. Draws polylines on the frame for two predefined areas (area1 and area2) using specified colors.
        2. Counts the number of entries and exits.
        3. Overlays text on the frame to display the count of entries and exits.
        """
        cv2.polylines(frame, [np.array(self.area1, np.int32)], True, color.area1(), 2)
        cv2.polylines(frame, [np.array(self.area2, np.int32)], True, color.area2(), 2)
        enter = len(self.entering)
        exit = len(self.exiting)
        cvzone.putTextRect(frame, str(f"Enter: {enter}"), (20, 30), 1, 1, color.text1(), color.text2())
        cvzone.putTextRect(frame, str(f"Exit: {exit}"), (20, 60), 1, 1, color.text1(), color.text2())

    # Main method to process the video
    def main(self):
        cap = cv2.VideoCapture(self.file_path)

        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        output_file_path = os.path.join(downloads_path, 'output_video.avi')
        out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'XVID'), 24.0, self.frame_size)

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame, self.frame_size)

                # detections = self.detect_object(frame)
                detections_person, detections_face = self.detect_BboxOnly(frame)
                self.counter(frame, detections_person, detections_face)

                out.write(frame)
                # self.show_time(frame)
                cv2.imshow(self.name_frame, frame)
                
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(self.name_frame, cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key == ord('p'):
                self.paused = not self.paused

        
            #if cv2.waitKey()&0xFF == ord('q'): break
            #if cv2.waitKey(0)&0xFF == 27: continue

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Return count of people entering and exiting
        result = {
            'total_people_entering': len(self.entering),
            'total_people_exiting': len(self.exiting),
            'entering_details': sorted([{'person_id': person[0], 'time': person[1].strftime('%Y-%m-%d %H:%M:%S')} for person in self.entering], key=lambda x: x['time']),
            'exiting_details': sorted([{'person_id': person[0], 'time': person[1].strftime('%Y-%m-%d %H:%M:%S')} for person in self.exiting], key=lambda x: x['time'])
        }
        return  result

if __name__ == '__main__':
    area1 = [(359, 559), (400, 559), (667, 675), (632, 681)]
    area2 = [(346, 563), (313, 566), (579, 703), (624, 694)]
    sample_video_path = 'Sample Test File\\test_video.mp4'
    frame_width = 1280
    frame_height = int(frame_width / 16 * 9)   
    algo = Algorithm_Count(sample_video_path, area1, area2, (frame_width, frame_height))
    r = algo.main()
    print(r)
