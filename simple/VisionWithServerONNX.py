import os
from flask import Flask, Response
import threading
import numpy as np
import cv2
import onnxruntime as ort
import time
from _utils import _improveLight
import pyrealsense2 as rs

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
            time.sleep(0.01)
            _, buffer = cv2.imencode('.jpg', current_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_flask():
    app.run(host='10.43.82.131', port=5000, debug=False, use_reloader=False)

flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

# ---- Simulated Depth Parameters ----
SIMULATED_DEPTH = 1.5  # meters
fx, fy = 600, 600
ppx, ppy = 320, 240

# ---- ONNX and Camera Setup ----
onnx_path = "best640lobb.onnx"
conf_thres = 0.1
iou_thres = 0.7
improvement_thresh = -0.7
cap = cv2.VideoCapture(0)

session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [o.name for o in session.get_outputs()]
class_names = {0: 'U', 1: 'R'}

# ---- Inference Helper ----
def preprocess(img):
    img = cv2.resize(img, (input_shape[3], input_shape[2]))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[None]  # CHW and add batch dim
    return img

def postprocess(outputs):
    return outputs[0]  # shape: (N, 6) or similar [x, y, w, h, conf, cls]


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


# ---- Main Loop ----
while True:

    frames = pipeline.wait_for_frames()
    frames = align_color.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("No frames received")
        continue

    img = _improveLight(np.asanyarray(color_frame.get_data()), improvement_thresh)
    input_tensor = preprocess(img)
    output = session.run(output_names, {input_name: input_tensor})[0]
    print(session.get_providers())
    # boxes = output[output[:, 4] > conf_thres]
    preds = np.squeeze(output[0])         # shape: (9, 8400)
    preds = preds.transpose()             # shape: (8400, 9)
    boxes = preds[preds[:, 4] > conf_thres]

    annotated_frame = np.asanyarray(color_frame.get_data()).copy()
    for box in boxes:
        x, y, w, h, conf, cls = box[:6]
        x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
        label = class_names.get(int(cls), 'Unknown')
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if label == 'R':
            u, v = int(x), int(y)
            x_3d = (u - ppx) * SIMULATED_DEPTH / fx
            y_3d = (v - ppy) * SIMULATED_DEPTH / fy
            print(f"class: {label}, 3D pos: ({x_3d:.2f}, {y_3d:.2f}, {SIMULATED_DEPTH:.2f})")

    # ---- FPS Calculation ----
    if 'prev_time' not in locals():
        prev_time = time.time()
    else:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (annotated_frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    current_frame = annotated_frame
    cv2.imshow("Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()