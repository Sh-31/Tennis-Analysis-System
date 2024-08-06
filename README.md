# Tennis-Analysis-System
---
Tennis-Analysis-System is a system that analyzes tennis matches and provides player statistics and mini court visualizations and It utilizes YOLO (You Only Look Once) v8x for player detection and tracking, and fine-tunes YOLOv5 for ball detection and tracking. Additionally, a ResNet50 model is used for keypoint detection and tracking.


## Usage
---
To install the Tennis-Analysis-system project, follow these steps:

1. Clone the repository:
```shell
git clone https://github.com/Sh-31/Tennis-Analysis-system.git
```
2. Install the required dependencies:
```shell
pip3 install -r requirements.txt (linux)
pip  install -r requirements.txt (windows)
```
3. Run the main script:
```shell
python3 main.py (linux)
python main.py (windows)
```

## Sample Output
---
<video width="600" controls>
  <source src="output_videos/output_video.mp4" type="video/mp4">
</video>

## Datasets
---
#### Ball Detection
For ball detection, we used the tennis-ball-detection dataset from Roboflow. You can access it at the following link:
- [Tennis Ball Detection - Roboflow](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection)

#### Keypoint Detection
For keypoint detection, we used a dataset collected by yastrebksv. You can download it from the link below:
- [TennisCourtDetector - Google Drive](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu/view?usp=drive_link)

You can also download it using the code provided in the `Tennis-Analysis-system\fine_tuning\explore.ipynb` notebook.

## Results
---



## Features
---
- Player detection and tracking using YOLOv8
- Ball detection and tracking using YOLOv5 (fine-tuned)
- Keypoint detection and tracking using a Resnet50 model (fine-tuned)
- Mini court visualizations with player and ball positions
- Player statistics calculation

## Limitations
---
- Mini-court coordination is not generic (It needs the true height of the player to convert the proportion of meters to pixels). 
