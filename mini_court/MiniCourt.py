import cv2
import sys 
import numpy as np
sys.path.append('../')
from utils import  *

SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48
PLAYER_1_HEIGHT_METERS = 1.88
PLAYER_2_HEIGHT_METERS = 1.91

class MiniCourt:
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50 # difference between the image (Original resolution) and the white rectangle
        self.padding_court = 20 # difference between the white rectangle and the actual tennis court

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy() 

        # frame.shape ---> heigth , width of org frame
        # x --> width of rectangle, y --> heigh of rectangle

        self.y_start = self.buffer
        self.y_end   = self.y_start + self.drawing_rectangle_height 

        self.x_end   = frame.shape[1] - self.buffer 
        self.x_start = self.x_end  - self.drawing_rectangle_width 
        
    def set_mini_court_position(self):

        self.court_x_start = self.x_start + self.padding_court
        self.court_y_start = self.y_start + self.padding_court
        self.court_x_end = self.x_end - self.padding_court
        self.court_y_end = self.y_end - self.padding_court

        self.court_drawing_width = self.court_x_end - self.court_x_start
        self.court_drawing_heigth = self.court_y_end - self.court_y_start

    def connvert_meters_pixels(self, meters):
         return covert_meters_to_pixal_distance(meters, DOUBLE_LINE_WIDTH, self.court_drawing_width)   

    def set_court_drawing_key_points(self):
        drawing_key_points = [0 for i in range(28) ]

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_x_start), int(self.court_y_start)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_x_end), int(self.court_y_start)
        # point 2
        drawing_key_points[4] = int(self.court_x_start)
        drawing_key_points[5] = self.court_y_start + self.connvert_meters_pixels(HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.connvert_meters_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # point 5
        drawing_key_points[10] = drawing_key_points[4] + self.connvert_meters_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # point 6
        drawing_key_points[12] = drawing_key_points[2] - self.connvert_meters_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # point 7
        drawing_key_points[14] = drawing_key_points[6] - self.connvert_meters_pixels(DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.connvert_meters_pixels(NO_MANS_LAND_HEIGHT)
        # point 9
        drawing_key_points[18] = drawing_key_points[16] + self.connvert_meters_pixels(SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.connvert_meters_pixels(NO_MANS_LAND_HEIGHT)
        # point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.connvert_meters_pixels(SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]     

    def draw_court(self, frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.x_start, self.y_start), (self.x_end, self.y_end), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask] # Transparency factor

        return out

    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_mini_court_coordinates(self, object_pos, keypoint_closest_pos, keypoint_index, player_height_in_pixels, player_id):
        object_x , object_y = object_pos
        keypoint_x_pos , keypoint_y_pos = keypoint_closest_pos

        dis_in_x = abs(object_x - keypoint_x_pos)
        dis_in_y = abs(object_y - keypoint_y_pos)

        # Convert from pixal distance to meter

        dis_in_x = convert_pixel_distance_to_meter(pixel_distance=dis_in_x, ref_heigth_meters=self.player_heights[player_id], ref_heigth_pixel=player_height_in_pixels)
        dis_in_y = convert_pixel_distance_to_meter(pixel_distance=dis_in_y, ref_heigth_meters=self.player_heights[player_id], ref_heigth_pixel=player_height_in_pixels)

        # Convert to mini court coordinates

        mini_court_x_distance_pixels = covert_meters_to_pixal_distance(dis_in_x, DOUBLE_LINE_WIDTH, self.court_drawing_width)   
        mini_court_y_distance_pixels = covert_meters_to_pixal_distance(dis_in_y, DOUBLE_LINE_WIDTH, self.court_drawing_width)   

        closest_mini_coourt_keypoint = (self.drawing_key_points[keypoint_index*2], self.drawing_key_points[keypoint_index*2+1])
        
        mini_court_object_position = (
                                      closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                     )

        return  mini_court_object_position


    def convert_bbox_to_mini_court_coordinates(self, player_detections , ball_detections , keypoints_org):

        self.player_heights = {
            1: PLAYER_1_HEIGHT_METERS,
            2: PLAYER_2_HEIGHT_METERS
        }

        player_bbox_mini = []
        ball_bbox_mini = []
    
        for frame_id , player_detection in enumerate(player_detections):

            ball_box = ball_detections[frame_id][1]
            ball_position = get_center_of_bbox(ball_box)
            closest_player_id_to_ball = min(player_detection.keys(), key=lambda x: L2_norm(ball_position, get_center_of_bbox(player_detection[x])))

            player_bbox_mini_dict = {}
            for player_id , bbox in player_detection.items():
                
                player_foot_pos = get_foot_position(bbox)

                closest_keypoint = get_closest_keypoint_index(player_foot_pos,keypoints_org, [0, 2, 12, 13])
                keypoint_closset_pos = (keypoints_org[closest_keypoint*2], 
                                     keypoints_org[closest_keypoint*2+1])

                # Get Player height in pixels
                frame_index_min = max(0, frame_id-20)
                frame_index_max = min(len(player_detections), frame_id+50)
                bboxes_heights_in_pixels = [get_height_of_bbox(player_detections[i][player_id]) for i in range (frame_index_min,frame_index_max)]
                max_player_height_in_pixels = max(bboxes_heights_in_pixels)
    
                mini_court_player_position = self.get_mini_court_coordinates(player_foot_pos, keypoint_closset_pos, closest_keypoint, max_player_height_in_pixels, player_id)
                player_bbox_mini_dict[player_id] = mini_court_player_position

                if closest_player_id_to_ball == player_id:
                    closest_keypoint = get_closest_keypoint_index(ball_position, keypoints_org, [0, 2, 12, 13])
                    keypoint_closset_pos = (keypoints_org[closest_keypoint*2], 
                                        keypoints_org[closest_keypoint*2+1])
                 
                    mini_court_ball_position = self.get_mini_court_coordinates( ball_position,
                                                                                  keypoint_closset_pos, 
                                                                                  closest_keypoint,
                                                                                  max_player_height_in_pixels, 
                                                                                  player_id
                                                                                 ) 
                    ball_bbox_mini.append({1:mini_court_ball_position})

            player_bbox_mini.append(player_bbox_mini_dict)                             
        
        return player_bbox_mini , ball_bbox_mini

    def draw_points_on_mini_court(self, frames, postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames



                
