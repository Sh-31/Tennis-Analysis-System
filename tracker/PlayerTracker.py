from ultralytics import YOLO
import cv2
import pickle
# to use utils 
import sys
sys.path.append('../')
from utils import mid_point , L2_norm


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results =  self.model.track(frame, persist=True)[0] # track uses tracking algorithms to assign unique IDs to detected objects and maintain their identities across frames. 
        name_id_dict = results.names # get dict of id -> class 

        player_dict = {}
        # iter for all bboxes in the frame
        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) # becase id of tracked object
            result = box.xyxy.tolist()[0] # get position
            object_cls_id = box.cls.tolist()[0] # get class_id
            object_cls_name = name_id_dict[object_cls_id] # map class_id to name of class
            if object_cls_name == "person": 
                player_dict[track_id] = result # if persion add to player_dict by track id
        
        return player_dict 

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections 

    def filter_players_from_other_people(self, player_detections, keypoints):
        # we will selecte players from other people by calculated L2 distance between court keypoints and selected less two distance
        player_detections_first_frame = player_detections[0] # one frame 
        distances = []

        for track_id, bbox in player_detections_first_frame.items():
            x1, y1, x2, y2 = bbox

            x = mid_point(x1, x2)
            y = mid_point(y1, y2)
            player_pos = (x, y)

            min_distance = float('inf')
            for i in range(0,len(keypoints),2):
                court_keypoint_pos = (keypoints[i], keypoints[i+1])
                distance = L2_norm(player_pos, court_keypoint_pos)
                if distance < min_distance:
                    min_distance = distance 
            distances.append((track_id, min_distance)) # we add just the closest court keypoint distance
               
        distances.sort(key = lambda x: x[1]) # sort by min_distance

        chosen_players = []
        for i in range(len(player_detections)): # loop in each frame
            chosen_players.append({
                distances[0][0]: player_detections[i][distances[0][0]],
                distances[1][0]: player_detections[i][distances[1][0]],
            }   
            )
        return chosen_players


    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
            output_video_frames.append(frame)
        
        return output_video_frames    
