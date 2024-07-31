import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    # downloading dataset

    # install('roboflow')
    # from roboflow import Roboflow

    # rf = Roboflow(api_key="tBjW2Qz3Fow3GC4iSbES")
    # project = rf.workspace("viren-dhanwani").project("tennis-ball-detection")
    # version = project.version(6)
    # dataset = version.download("yolov5")

    # fine-tuing YOLO for ball detection
    root_path = "/teamspace/studios/this_studio/Tennis-Analysis-system/fine_tuning"

    command = [
    'yolo',
    'task=detect',
    'mode=train',
    'model=yolov5l6u.pt',
    f'data={root_path}/data/tennis-ball-detection/data.yaml',
    'epochs=100',
    'imgsz=640'
]

    subprocess.run(command) # same as !yolo task=detect mode=train model=yolov5l6u.pt data="/data/tennis-ball-detection-6/data.yaml" epochs=100 imgsz=640

