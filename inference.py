from ultralytics import YOLO 


if __name__ == "__main__":
    model = YOLO('yolov8x')

    result = model.track('sample_data/input_video.mp4', conf=0.2, save=True, )

    print(result)
    print("boxes:")
    for box in result[0].boxes:
        print(box)