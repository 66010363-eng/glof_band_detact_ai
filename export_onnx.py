from ultralytics import YOLO

model = YOLO("my_model.pt")

model.export(
    format="onnx",
    opset=12,
    simplify=True,
    dynamic=False
)
