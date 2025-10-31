# from ultralytics import YOLO
# model = YOLO("best640lobb.pt")
# model.export(format="onnx", opset=12, task="obb", dynamic=False, imgsz=640, name="640bestobb_correct")

# from ultralytics import YOLO

# model = YOLO("best640lobb.pt")
# print(model.model.nc)       # Number of classes
# print(model.names)          # Class names (dict: {0: 'class0', 1: 'class1', ...})


from ultralytics import YOLO
import numpy as np

model = YOLO("best640lobb_myTRT.engine", task="obb")

# Run dummy prediction to initialize internal structures (and force error)
try:
    model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), save=False)
except RuntimeError as e:
    pass  # we just want to trigger model loading

# ✅ Patch the internal number of classes and names
backend_model = model.predictor.model
backend_model.nc = 4
backend_model.names = {
    0: "U",
    1: "R",
    2: "Strawberry",
    3: "Blackberry"
}

# ✅ Now run inference (should work without crash)
for result in model(source=0, stream=True):
    result.plot()


# import onnxruntime as ort
# import numpy as np

# session = ort.InferenceSession("best640lobb.onnx")
# input_shape = session.get_inputs()[0].shape
# dummy = np.zeros(input_shape, dtype=np.float32)
# output = session.run(None, {session.get_inputs()[0].name: dummy})[0]
# print("ONNX output shape:", output.shape)