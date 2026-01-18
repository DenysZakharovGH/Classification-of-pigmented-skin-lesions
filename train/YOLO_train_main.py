import torch
from ultralytics import YOLO

from train.config import YOLO_settings_file_path

# Load the model.
model = YOLO('yolov8s-cls.pt')
#model = YOLO('/home/denys/PycharmProjects/Classification-of-pigmented-skin-lesions/runs/detect/train6/weights/last.pt')

# Training.
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))




# Training.
results = model.train(
    data=YOLO_settings_file_path,
    imgsz=224,          # ↓ reduce image size
    epochs=50,
    batch=64,            # ↓ reduce batch
    save_period=5,
)

#yolo classify train model=yolov8s-cls.pt imgsz=224 batch=64 epochs=50