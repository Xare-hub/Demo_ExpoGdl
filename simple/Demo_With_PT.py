import cv2
from ultralytics import YOLO
import pyrealsense2 as rs
import time


# Load model
model = YOLO("best640lobb.pt", task="obb")  # replace with your .pt path
# model = YOLO("best640lobb_myTRT.engine", task="obb")  # replace with your .pt path

class_names = {0: 'U', 1: 'R'}

# Video capture
cap = cv2.VideoCapture(0)
conf_thres = 0.25

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



while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame, imgsz=640, conf=conf_thres)[0]

    if results.obb is not None and len(results.obb) > 0:
        for obb in results.obb:
            poly = obb.xyxyxyxy.cpu().numpy().astype(int).reshape(-1, 1, 2)
            cls = int(obb.cls.item())
            conf = float(obb.conf.item())
            label = f"{class_names.get(cls, str(cls))} {conf:.2f}"

            cv2.polylines(frame, [poly], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, label, tuple(poly[0][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---- FPS Calculation ----
    if 'prev_time' not in locals():
        prev_time = time.time()
    else:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv8-OBB .pt Prediction", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
