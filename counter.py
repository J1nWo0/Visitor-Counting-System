import os
import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
from tracker import *


class Color:
    def boundingBox1(self):
        green = (0,255,0)
        return green
    def boundingBox2(self):
        yellow = (0,255,255)
        return yellow
    def text1(self):
        white = (255,255,255)
        return white
    def text2(self):
        black = (0,0,0)
        return black
    def area1(self):
        blue = (255,0,0)
        return blue
    def area2(self):
        red = (0, 0, 255)
        return red
    def point(self):
        pink = (255,0,255)
        return pink 
    def center_point(self):
        cyan = (255,255,0)
        return cyan
    def rectangle(self):
        orange = (0,119,255)
        return orange


color = Color()
tracker = Tracker()
model=YOLO('yolo-Weights\yolov8s-seg.pt')


class Algorithm_Count:
    def __init__(self, a1, a2):
        self.peopleEntering = {}
        self.entering = set()
        self.peopleExiting = {}
        self.exiting = set()
        self.area1 = a1
        self.area2 = a2
        self.paused = False
        self.coordinates = []
        self.start_time = time.time()

        cv2.namedWindow('Frame')


    def detect_BboxOnly(self, frame):
        results = model(frame, conf=0.6, classes=[0])
        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                detections.append([int(x1), int(y1), int(x2), int(y2), float(score)])
        return detections
    
    def detect_withSegments(self, frame):
        height, width, channels = frame.shape

        results = self.model.predict(source=frame.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        for seg in result.masks.xyn:
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores

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
        list = []
        for box in detections:
            if len(box) == 5:
                x1, y1, x2, y2, score = box
                class_id = 0  # Default class ID for a person
            elif len(box) == 6:
                x1, y1, x2, y2, score, class_id = box
            else: continue  # Skip boxes with unexpected length

            if class_id == 0:  # Assuming person class is 0
                # label = f"Person: {score:.2f}"
                list.append([x1, y1, x2, y2])

        # counting the number of detected objects/person
        bbox_id = tracker.update(list)

        for bbox in bbox_id:
            x1, y1, x2, y2, id = bbox
            label = f"{id} Person: {score:.2f}"

            self.person_bounding_boxes(frame, x1, y1, x2, y2, id)
            self.people_entering(frame, x1, y1, x2, y2, id, label)
            self.people_exiting(frame, x1, y1, x2, y2, id, label)

        self.draw_polylines(frame)


    def person_bounding_boxes(self, frame, x1, y1, x2, y2, id):
        if id != -1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color.rectangle(), 2)

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


    def main(self, video_path):
        cap = cv2.VideoCapture(video_path)

        downloads_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        output_file_path = os.path.join(downloads_path, 'output_video.avi')
        out = cv2.VideoWriter(output_file_path,cv2.VideoWriter_fourcc(*'XVID'), 24.0, (1020,500))

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.resize(frame,(1020,500))

                #results = model.track(frame, persist=True, conf=0.5)
                #frame_ = results[0].plot()

                detections = self.detect_BboxOnly(frame)
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



if __name__ == '__main__':
    area1 = [(312, 388), (289, 390), (474, 469), (497, 462)]
    area2 = [(279, 392), (250, 397), (423, 477), (454, 469)]
    sample_video_path = 'Sample Test File\\test_video.mp4'
    algo = Algorithm_Count(area1, area2)
    algo.main(sample_video_path)