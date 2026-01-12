from ultralytics import YOLO

from app.app_config import settings

model = YOLO(settings.cnn.model_path)