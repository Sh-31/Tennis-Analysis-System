import cv2
import sys 
import numpy as np
sys.path.append('../')
from utils import *

SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48

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

    def  connvert_meters_pixels(self, meters):
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
    
