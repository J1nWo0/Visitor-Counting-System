import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
from tracker import *
import torch


class Color:
    def __init__(self):
        self.colors = {
            'boundingBox1': (0, 255, 0),
            'boundingBox2': (0, 255, 255),
            'text1': (255, 255, 255),
            'text2': (0, 0, 0),
            'area1': (255, 0, 0),
            'area2': (0, 0, 255),
            'point': (255, 0, 255),
            'center_point': (255, 255, 0),
            'rectangle': (0, 119, 255)
        }

    def __getattr__(self, item):
        return lambda: self.colors[item]

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

color = Color()
tracker = Tracker()
model=YOLO('yolo-Weights\\yolo11n-seg.pt').to(device)



class Algorithm_Count:
    def __init__(self, file_path, a1, a2, frame_size):
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
        self.start_time = time.time()

        cv2.namedWindow('Frame')


    def detect_object(self, frame):
        results = model.track(frame, conf=0.6, classes=[i for i in range(0, 80)], persist=True, tracker="bytetrack.yaml")

        detections = []
        for r in results:  # Iterate through the results
            boxes = r.boxes  # Extract bounding boxes
            masks = r.masks  # Extract segmentation masks (if present)
            
            for i, box in enumerate(boxes):
                # Extracting bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]  # Get the top-left (x1, y1) and bottom-right (x2, y2) coordinates
                # Extracting the tracking ID (if present)
                id = box.id if box.id is not None else -1  # ID will be -1 if there's no ID assigned
                # Extracting confidence score
                score = box.conf[0] if box.conf is not None else 0.0  # Default to 0.0 if confidence is missing
                # Extract segmentation mask if available (convert mask to numpy array)
                mask = np.array(masks[i].xy, dtype=np.int32) if masks is not None else None  # Convert mask to numpy array if it exists
                
                # # Append to the detections list as a tuple
                # detections.append({
                #     "id": int(id),  # Tracking ID (or -1 if not tracked)
                #     "x1": int(x1),  # Top-left x-coordinate
                #     "y1": int(y1),  # Top-left y-coordinate
                #     "x2": int(x2),  # Bottom-right x-coordinate
                #     "y2": int(y2)   # Bottom-right y-coordinate
                # })
                detections.append([int(x1), int(y1), int(x2), int(y2), int(id), float(score), mask])
        
        return detections
    
    def show_time(self, frame):
        elapsed_time = time.time() - self.start_time

        # Convert elapsed time to hours, minutes, seconds, and milliseconds
        milliseconds = int(elapsed_time * 1000) #/ 6.0001
        hours, milliseconds = divmod(milliseconds, 3600000)
        minutes, milliseconds = divmod(milliseconds, 60000)
        seconds = (milliseconds / 1000.0)

        # Display the time in the format "hour:minute:second.millisecond"
        time_str = "Running Time: {:02}:{:02}:{:06.3f}".format(int(hours), int(minutes), seconds)
        cvzone.putTextRect(frame,time_str, (20,480), 1,1, color.text1(), color.text2())

    def counter(self, frame, detections):

        for detect in detections:
            x1, y1, x2, y2, id, score, mask = detect
            label = f"{id} Person: {score:.2f}"
            

            self.person_bounding_boxes(frame, x1, y1, x2, y2, id, score, mask)
            self.people_entering(frame, x1, y1, x2, y2, id, label)
            self.people_exiting(frame, x1, y1, x2, y2, id, label)
            

        self.draw_polylines(frame)

    def person_bounding_boxes(self, frame, x1, y1, x2, y2, id, score, mask):
        if id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.rectangle(), 2)
            cvzone.putTextRect(frame,  f"{id}: {score:.2f}", (x1, y1 - 10), 1,1, color.text1(), color.text2())
            cv2.circle(frame, (x2, y2), 4, color.point(), -1)  
            # Check if mask is valid and draw it
            if mask is not None:
                # cv2.fillPoly(frame, [mask], color.center_point()) # Fill the mask with a color
                cv2.polylines(frame, [mask], True, color.center_point(), 2)  # Draw the mask outline

    def people_entering(self, frame, x1, y1, x2, y2, id, label):
        result_p1 = cv2.pointPolygonTest(np.array(self.area2,np.int32), ((x2,y2)), False)
        if result_p1 >= 0:
            self.peopleEntering[id] = (x2, y2) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox2(), 2)
            cvzone.putTextRect(frame, label, (x1+10, y1-10), 1,1, color.text1(), color.text2()) 
            #cv2.putText(frame, label2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.text1(), 2)
        if id in self.peopleEntering:
            result_p2 = cv2.pointPolygonTest(np.array(self.area1,np.int32), ((x2,y2)), False)
            if result_p2 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox1(), 2)
                cv2.circle(frame, (x2, y2), 4, color.point(), -1)  
                #cv2.putText(frame, label2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.text1(), 2)
                cvzone.putTextRect(frame, label, (x1+10, y1-10), 1,1, color.text1(), color.text2())
                self.entering.add(id)

    def people_exiting(self, frame, x1, y1, x2, y2, id, label):
        result_p3 = cv2.pointPolygonTest(np.array(self.area1,np.int32), ((x2,y2)), False)
        if result_p3 >= 0:
            self.peopleExiting[id] = (x2, y2) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox1(), 2)
            cvzone.putTextRect(frame, label, (x1+10, y1-10), 1,1, color.text1(), color.text2()) 
        if id in self.peopleExiting:
            result_p4 = cv2.pointPolygonTest(np.array(self.area2,np.int32), ((x2,y2)), False)
            if result_p4 >= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.boundingBox2(), 2)
                cv2.circle(frame, (x2, y2), 4, color.point(), -1)  
                #cv2.putText(frame, label2, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.text1(), 2)
                cvzone.putTextRect(frame, label, (x1+10, y1-10), 1,1, color.text1(), color.text2())
                self.exiting.add(id)

    def draw_polylines(self, frame):
        cv2.polylines(frame,[np.array(self.area1,np.int32)],True,color.area1(),2)
        #cvzone.putTextRect(frame,str('1'), (self.area1[3][0]+5, self.area1[3][1]+2), 1,1, color.text1(), color.text2())

        cv2.polylines(frame,[np.array(self.area2,np.int32)],True,color.area2(),2)
        #cvzone.putTextRect(frame,str('2'), (self.area2[3][0]+5, self.area2[3][1]+2), 1,1, color.text1(), color.text2())
        enter = len(self.entering)
        exit = len(self.exiting)
        cvzone.putTextRect(frame,str(f"Enter: {enter}"), (20,30), 1,1, color.text1(), color.text2())
        cvzone.putTextRect(frame,str(f"Exit: {exit}"), (20,60), 1,1, color.text1(), color.text2())


    def main(self):
        cap = cv2.VideoCapture(self.file_path)

        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        output_file_path = os.path.join(downloads_path, 'output_video.avi')
        out = cv2.VideoWriter(output_file_path,cv2.VideoWriter_fourcc(*'XVID'), 24.0, self.frame_size)

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame,self.frame_size)

                #results = model.track(frame, persist=True, conf=0.5)
                #frame_ = results[0].plot()

                detections = self.detect_object(frame)
                self.counter(frame, detections)

                out.write(frame)
                # self.show_time(frame)
                cv2.imshow('Frame', frame)
                

            key = cv2.waitKey(1)&0xFF
            if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1: 
                break
            elif key == ord('p'):
                self.paused = not self.paused

        
            #if cv2.waitKey()&0xFF == ord('q'): break
            #if cv2.waitKey(0)&0xFF == 27: continue

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # return date of people entering and exiting
        return [len(self.entering), len(self.exiting)]



if __name__ == '__main__':
    area1 = [(359, 559), (400, 559), (667, 675), (632, 681)]
    area2 = [(346, 563), (313, 566), (579, 703), (624, 694)]
    sample_video_path = 'Sample Test File\\test_video.mp4'
    frame_width = 1280
    frame_height = int(frame_width / 16 * 9)   
    algo = Algorithm_Count(sample_video_path, area1, area2, (frame_width, frame_height))
    r=algo.main()
    print(r)