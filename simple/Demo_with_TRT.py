import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

TRT_LOGGER = trt.Logger()

# Load engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate buffers
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, h_output, d_input, d_output

# Run inference
def infer(context, bindings, d_input, d_output, h_input, h_output):
    cuda.memcpy_htod(d_input, h_input)
    context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

# Load engine and context
engine = load_engine("best640lobb_myTRT.engine")
context = engine.create_execution_context()
h_input, h_output, d_input, d_output = allocate_buffers(engine)
bindings = [int(d_input), int(d_output)]

# Camera capture
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess (resize + normalize)
    input_img = cv2.resize(frame, (640, 640))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = input_img.transpose(2, 0, 1).ravel()  # CHW
    np.copyto(h_input, input_img)

    # Inference
    output = infer(context, bindings, d_input, d_output, h_input, h_output)

    # Postprocess
    output = output.reshape(1, 9, 8400)      # reshape raw flat output
    preds = np.squeeze(output, axis=0).T     # shape: (8400, 9)

    conf_thres = 0.25
    boxes = preds[preds[:, 4] > conf_thres]  # filter by confidence

    for box in boxes:
        x, y, w, h, conf, cls = box[:6]      # ignore extra entries if present
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = f"{int(cls)} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---- FPS Calculation ----
    if 'prev_time' not in locals():
        prev_time = time.time()
    else:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    # print("Output:", output[:10])  # Example output view

    cv2.imshow("Cam", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
