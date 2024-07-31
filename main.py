from utils.video_utils import read_video, save_video
from tracker.PlayerTracker import PlayerTracker
from tracker.BallTracker import BallTracker
from tracker.KeypointsTracer import KeypointsDetector
import cv2
import pandas as pd


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
    # ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
   
    keypoints_model_path = "models/keypoints/model_last.pth"
    keypoints_detector = KeypointsDetector(model_path=keypoints_model_path)
    keypoints = keypoints_detector.predict(video_frames[0])

    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    output_video_frames = keypoints_detector.draw_keypoints_on_video(output_video_frames, keypoints)
    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()