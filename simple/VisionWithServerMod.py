import os
from flask import Flask, Response
import threading
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from statistics import mean
import time
from _utils import _improveLight

# ---- Flask App ----
app = Flask(__name__)
current_frame = None

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global current_frame
    while True:
        if current_frame is not None:
            time.sleep(0.1)
            _, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_flask():
    app.run(host='10.43.82.131', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# ---- Object Class ----
class fruitPos:
    def __init__(self, x, y, z, cl):
        self.x = x
        self.y = y
        self.z = z
        self.cl = cl

# ---- Simulated Depth Parameters ----
SIMULATED_DEPTH = 1.5  # meters
fx, fy = 600, 600
ppx, ppy = 320, 240

# ---- YOLO and Camera Setup ----
conf_thres = 0.1
iou_thres = 0.7
improvement_thresh = -0.7


cap = cv2.VideoCapture(0)

# ---- YOLO Engine Setup ----
model = YOLO("best640lobb_myTRT.engine", task="obb")

# Run dummy prediction to initialize internal predictor
try:
    model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), save=False)
except Exception:
    pass  # Expected error due to shape mismatch; we just want model.predictor ready

# ✅ Patch class count and names
backend_model = model.predictor.model
backend_model.nc = 4
backend_model.names = {
    0: "U",
    1: "R",
    2: "Strawberry",
    3: "Blackberry"
}


# ---- Main Loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    img = _improveLight(frame, improvement_thresh)
    pred = model(img, verbose=False, conf=conf_thres, iou=iou_thres, imgsz=640)
    results = pred[0]
    annotated_frame = pred[0].plot()
    iterate = results.obb if model.task == 'obb' else results.boxes

    fruits = []
    for box in iterate:
        if model.task == 'obb':
            inftensor = box.xywhr.cpu().data.numpy()
        else:
            inftensor = box.xywh.cpu().data.numpy()
        u = round(inftensor[0][0])
        v = round(inftensor[0][1])
        distance = SIMULATED_DEPTH

        if results.names[int(box.cls)] == "R":
            x = (u - ppx) * distance / fx
            y = (v - ppy) * distance / fy
            z = distance
            fruit = fruitPos(x, y, z, results.names[int(box.cls)])
            print(f"class: {fruit.cl}, 3D pos: ({fruit.x:.2f}, {fruit.y:.2f}, {fruit.z:.2f})")
            fruits.append(fruit)

    current_frame = annotated_frame
    cv2.imshow("Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()



