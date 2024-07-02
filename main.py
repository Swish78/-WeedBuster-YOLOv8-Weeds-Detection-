from ultralytics import YOLO


model = YOLO("yolov8n.pt")

model.train(data="weed_detection.yaml", epochs=10)
metrics = model.val()
path = model.export(format="onnx")