from counter import Algorithm_Count
from set_coordinates import ClickPoints 

class VideoProcessor:
    def __init__(self, in_video_path, frame_width, area1_coordinates=None, area2_coordinates=None):
        self.in_video_path = in_video_path
        self.area1_coordinates = area1_coordinates
        self.area2_coordinates = area2_coordinates
        # frame_width = 1280
        frame_height = int(frame_width / 16 * 9)    
        self.frame_size = (frame_width, frame_height)

    def get_coordinates(self, current_coordinates, other_coordinates, area):
        if not current_coordinates:
            click_points = ClickPoints(self.in_video_path, other_coordinates)
            current_coordinates = click_points.run(self.frame_size)

        if not current_coordinates:
            print(f"Area {area} No coordinates")
            exit()

        if len(current_coordinates) < 4:
            print(f"Area {area} Incomplete")
            exit()

        return current_coordinates

    def process_video(self):
        self.area1_coordinates = self.get_coordinates(self.area1_coordinates, self.area2_coordinates, 1)
        self.area2_coordinates = self.get_coordinates(self.area2_coordinates, self.area1_coordinates, 2)

        print("Coordinates from ClickPoints (Area 1):", self.area1_coordinates)
        print("Coordinates from ClickPoints (Area 2):", self.area2_coordinates)

        algo = Algorithm_Count(self.in_video_path, self.area1_coordinates, self.area2_coordinates, self.frame_size)
        r = algo.main()
        print(r)

if __name__ == "__main__":
    a1 = [] #[(312,388),(289,390),(474,469),(497,462)]
    a2 = []
    in_video_path = 0#"Sample Test File\\test_video.mp4"    

    video_processor = VideoProcessor(in_video_path, 1280, a1, a2)
    video_processor.process_video()

    # Uncomment the following block if you want to use the webcam
    '''
    webcam_processor = Algorithm_Detection()
    webcam_processor.detectPeople()
    '''
    # example of the output
    # Coordinates from ClickPoints (Area 1): [(312, 388), (289, 390), (474, 469), (497, 462)]
