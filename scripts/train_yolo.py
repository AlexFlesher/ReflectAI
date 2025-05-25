#-------------------------------Imports--------------------------------------
from ultralytics import YOLO
#----------------------------------------------------------------------------

model = YOLO("yolov8n_trained1.pt")  # or 'yolov8s.pt' etc.
model.train(data="data.yaml", epochs=50, imgsz=640)
