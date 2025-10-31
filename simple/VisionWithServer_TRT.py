import os
from flask import Flask, Response, render_template, jsonify
import threading
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
from _utils import _improveLight
import pyrealsense2 as rs

app = Flask(__name__)
current_frame = None

@app.route('/')            
def index():
    return render_template('index.html')

current_frame = None  # Variable to hold the current frame for streaming
@app.route('/video_feed')
def video_feed():
    """Returns the current frame in a stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def get_counts():
    # Use a global or shared variable if needed — here it's recomputed
    counts = {
        "UnripeBlackberry": fruit_count["Unripe Blackberry"],
        "UnripeStrawberry": fruit_count["Unripe Strawberry"],
        "RipeBlackberry": fruit_count["Ripe Blackberry"],
        "RipeStrawberry": fruit_count["Ripe Strawberry"]
    }
    return jsonify(counts)


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
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

class fruitPos:
    def __init__(self, x, y, z, cl):
        self.x = x
        self.y = y
        self.z = z
        self.cl = cl

# ----- Simulated Camera Intrinsics -----
fx, fy = 600, 600
ppx, ppy = 320, 240
SIMULATED_DEPTH = 1.5  # meters

# ----- Model Config -----
weights = "best640lobb_myTRT.engine"
conf_thres = 0.50
iou_thres = 0.7
improvement_thresh = -0.7

# ----- Use OpenCV webcam -----
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise RuntimeError("Could not open webcam")


# ----- Use Intel Realsense webcam -----
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,30)
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8,30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    #filling = rs.hole_filling_filter(1)

align_to = rs.stream.color
align_color = rs.align(align_to)


model = YOLO("best640lobb_myTRT.engine", task="obb")

# Run dummy prediction to initialize internal structures (and force error)
try:
    model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), save=False)
except RuntimeError as e:
    pass  # we just want to trigger model loading

# Patch the internal number of classes and names
backend_model = model.predictor.model
backend_model.nc = 4
backend_model.names = {
    0: "Ripe Blackberry",                                #U
    1: "Ripe Strawberry",                                         #R
    2: "Unripe Blackberry",                                #Strawberry
    3: "Unripe Strawberry"                                 #Blackberry
}


# model = YOLO(weights, task='obb')
# model.names[0] = "U"
# model.names[1] = "R"
# names = model.names


# print(model.model.input_shapes)

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     print("Failed to capture frame")
    #     continue

    frames = pipeline.wait_for_frames()
    frames = align_color.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("No frames received")
        continue


    img = _improveLight(np.asanyarray(color_frame.get_data()), improvement_thresh)

    # img = cv2.resize(img, (640, 640))

    pred = model(img, verbose=False, conf=conf_thres, iou=iou_thres, imgsz=640)
    results = pred[0]
    annotated_frame = results.plot(line_width = 1, font_size = 4)
    iterate = results.obb if model.task == 'obb' else results.boxes

    fruits = []
    for box in iterate:
        inftensor = (box.xywhr if model.task == 'obb' else box.xywh).cpu().data.numpy()
        u = round(inftensor[0][0])
        v = round(inftensor[0][1])
        distance = SIMULATED_DEPTH

        print(f"class = {results.names[int(box.cls)]}, coordinates = {inftensor}, depth: {distance:.2f}")

        if results.names[int(box.cls)] == "R":
            x = (u - ppx) * distance / fx
            y = (v - ppy) * distance / fy
            z = distance
            fruit = fruitPos(x, y, z, results.names[int(box.cls)])
            print("3D position:", (x, y, z))
            fruits.append(fruit)

    fruit_count = {
    "Ripe Blackberry": 0,
    "Ripe Strawberry": 0,
    "Unripe Blackberry": 0,
    "Unripe Strawberry": 0
    }

    # Just before plotting each frame
    fruit_count = {k: 0 for k in fruit_count}  # Reset counts

    for box in iterate:
        cls = results.names[int(box.cls)]
        if cls in fruit_count:
            fruit_count[cls] += 1


    # ---- FPS Calculation ----
    if 'prev_time' not in locals():
        prev_time = time.time()
    else:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (annotated_frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Count: {len(iterate)}", (annotated_frame.shape[1] - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Inference", annotated_frame)
    current_frame = annotated_frame
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
