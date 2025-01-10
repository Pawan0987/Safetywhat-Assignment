import os

class Config:
    YOLO_WEIGHTS = "yolov5s.pt"
    YOLO_CONFIG = "yolov5s.yaml"
    VIDEO_PATH = "mumbai_traffic.mp4"
    OUTPUT_PATH = "output"

    @classmethod
    def create_output_directory(cls):
        if not os.path.exists(cls.OUTPUT_PATH):
            os.makedirs(cls.OUTPUT_PATH)

