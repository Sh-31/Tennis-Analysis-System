import cv2
import pandas as pd
from utils import read_video , save_video , calculate_player_statistics , draw_player_stats
from tracker import PlayerTracker , BallTracker , KeypointsDetector
from mini_court import MiniCourt

def main():
    # Read Video
    input_video_path = "sample_data/input_video.mp4"
    video_frames = read_video(input_video_path)
  
    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    ball_tracker = BallTracker(model_path='models/ball_detector/best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
   
    # Load keypoints model and inference 
    keypoints_model_path = "models/keypoints/model_last.pth"
    keypoints_detector = KeypointsDetector(model_path=keypoints_model_path)
    keypoints = keypoints_detector.predict(video_frames[0])

    # Select Player from other People 
    player_detections = player_tracker.filter_players_from_other_people(player_detections, keypoints)

    # get ball shots frames
    ball_shots_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    
    # draw mini court
    Mini_Court = MiniCourt(video_frames[0])
    output_video_frames = Mini_Court.draw_mini_court(video_frames)

    # draw player and ball in mini court
    player_mini_court_detections , ball_mini_court_detections = Mini_Court.convert_bbox_to_mini_court_coordinates(player_detections, ball_detections, keypoints)
    output_video_frames = Mini_Court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = Mini_Court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # calculate and draw player statistics
    player_stat = calculate_player_statistics(ball_shots_frames=ball_shots_frames, 
                                ball_mini_court_detections=ball_mini_court_detections,
                                player_mini_court_detections=player_mini_court_detections,
                                video_frames_number=len(video_frames))

    output_video_frames = draw_player_stats(output_video_frames, player_stat)                            

    # Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = keypoints_detector.draw_keypoints_on_video(output_video_frames, keypoints)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()