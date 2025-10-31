import os
import pyrealsense2 as rs
import cv2
import time
import numpy as np


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

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align_color.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame = aligned_frames.get_infrared_frame()

        if not color_frame or not depth_frame or not ir_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
            cv2.COLORMAP_JET
        )
        ir_image = np.asanyarray(ir_frame.get_data())

        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_colormap)
        cv2.imshow('Infrared', ir_image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
