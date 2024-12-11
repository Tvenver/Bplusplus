from ultralytics import YOLO
from pathlib import Path    

def train(input_yaml: str, output_directory: str = Path(__file__).resolve().parent, epochs: int = 30, imgsz: int = 640, batch: int = 16):
    model = YOLO('yolov8n.pt')
    model.train(data=input_yaml, epochs=epochs , imgsz=imgsz, batch=batch, project=output_directory)
    return model

def validate(model: YOLO, input_yaml: str, output_directory: str = Path(__file__).resolve().parent):
    metrics = model.val(data=input_yaml, project=output_directory)
    return metrics