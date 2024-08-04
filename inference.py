from ultralytics import YOLO 
import cv2

if __name__ == "__main__":
    # # Initialize the model
    # model = YOLO('yolov8x')

    # # Run tracking on the input video
    # result = model.track('sample_data/input_video.mp4', conf=0.2, save=True)

    # # Print results
    # print(result)
    # print("boxes:")
    # for box in result[0].boxes:
    #     print(box)

    import ffmpeg

    input_file = 'output_videos/output_video.avi'
    output_file = 'output_videos/output_video.mp4'

    ffmpeg.input(input_file).output(output_file).run()
