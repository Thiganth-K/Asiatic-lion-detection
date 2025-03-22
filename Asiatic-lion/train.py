from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load YOLOv8 model
model.train(data="C:/Users/THIGANTH/Asiatic-lion/datasets/data.yaml", epochs=5, imgsz=320, device="cpu")
