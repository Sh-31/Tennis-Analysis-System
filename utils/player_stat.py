import numpy as np
import cv2
import pandas as pd
from copy import deepcopy


from .bbox_utils import L2_norm
from .conversions import convert_pixel_distance_to_meter


def calculate_player_statistics(ball_shots_frames, ball_mini_court_detections, player_mini_court_detections, video_frames_number):

    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_1_average_shot_speed':0,
        'player_1_average_shot_speed':0,   

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
        'player_2_average_shot_speed':0,
        'player_2_average_shot_speed':0, 
      
    }]  # List of Dict (to store all player stat by frame)

    for i in range(len(ball_shots_frames) - 1):

        start_frame = ball_shots_frames[i]
        end_frame = ball_shots_frames[i + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24 frames per second

        distance_covered_by_ball_pixels = L2_norm(ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meter(
            distance_covered_by_ball_pixels,
            10.97,  # DOUBLE_LINE_WIDTH
            250     # rectangle width of mini court
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6  # convert meters/second to kilometers/hour

        # Identify the player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: L2_norm(player_positions[player_id], ball_mini_court_detections[start_frame][1]))

        # Calculate the opponent player's speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = L2_norm(
            player_mini_court_detections[start_frame][opponent_player_id], 
            player_mini_court_detections[end_frame][opponent_player_id]
        )

        distance_covered_by_opponent_meters = convert_pixel_distance_to_meter(
            distance_covered_by_opponent_pixels,
            10.97,  # DOUBLE_LINE_WIDTH
            250     # rectangle width of mini court
        )

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # add stat of last shot   
    current_player_stats = deepcopy(player_stats_data[-1])
    last_frame_index = ball_shots_frames[len(ball_shots_frames) - 1]
    current_player_stats['frame_num'] = last_frame_index
    player_positions = player_mini_court_detections[last_frame_index]
    player_shot_ball = min(player_positions.keys(), key=lambda player_id: L2_norm(player_positions[player_id], ball_mini_court_detections[last_frame_index][1]))
    current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1 
    player_stats_data.append(current_player_stats)


    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(video_frames_number))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill() # full frame bettween each shot
   
    # Calculate average speeds and replace NaNs with 0
    player_stats_data_df['player_1_average_shot_speed'] = (player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']).fillna(0)
    player_stats_data_df['player_2_average_shot_speed'] = (player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']).fillna(0)
    player_stats_data_df['player_1_average_player_speed'] = (player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']).fillna(0)
    player_stats_data_df['player_2_average_player_speed'] = (player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']).fillna(0)

    return player_stats_data_df

def check_nan_inf_avg_feature(feature):
    if feature in [np.nan, np.inf, -np.inf]:  
        return 0
    return feature


def draw_player_stats(output_video_frames, player_stats):

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']
        player_1_shot_number = row['player_1_number_of_shots']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']
        player_2_shot_number = row['player_2_number_of_shots']

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width = 350
        height = 250

        start_x = frame.shape[1] - 400
        start_y = frame.shape[0] - 500
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 80, start_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        text = "Shot Counter"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{int(player_1_shot_number)}              {int(player_2_shot_number)}"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 150, start_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        avg_player_1_shot_speed = check_nan_inf_avg_feature(avg_player_1_shot_speed)
        avg_player_2_shot_speed = check_nan_inf_avg_feature(avg_player_2_shot_speed)

        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 10, start_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        avg_player_1_speed = check_nan_inf_avg_feature(avg_player_1_speed)
        avg_player_2_speed = check_nan_inf_avg_feature(avg_player_2_speed)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x + 130, start_y + 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_video_frames


