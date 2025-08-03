import sys
from pathlib import Path

# Add yolov5/ to the Python path
YOLOV5_PATH = Path(__file__).parent.parent / "yolov5"
sys.path.append(str(YOLOV5_PATH))

from detect import run

run(
    weights="best.pt",       
    source="drone.jpg",      
    imgsz=(640, 640),
    conf_thres=0.25,
    save_txt=False,
    save_conf=True,
    project="inference_results",
    name="run1",
    exist_ok=True
)
