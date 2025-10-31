
import os

import pyrealsense2 as rs
import numpy as np
import cv2
import sys
from pathlib import Path
import matplotlib as plt
import cv2
import torch
import rospy
import std_msgs.msg
from ultralytics import YOLO
import cv2
from statistics import mean
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header
import time
from _utils import _improveLight


class fruitPos:
    def __init__(self, x, y, z, cl):
        self.x = x
        self.y = y
        self.z = z
        self.cl = cl

def getmeanDistance(w,D,x,y):
    try:
        data = []
        pad = int((w-1)/2)
        c = 0
        for i in range(x-pad, x+pad):
            for j in range(y-pad, y+pad):    
                if D.get_distance(i,j) > 0:
                    data.append(D.get_distance(i,j)) 
        return mean(data)
    except:
        return 0
def callback(msg):
    print(msg)
    
#os.environ['ROS_MASTER_URI'] = 'http://10.43.116.24:11311'
#os.environ['ROS_MASTER_URI'] = 'http://10.43.123.228:11311'
#os.environ['ROS_MASTER_URI'] = 'http://10.43.125.232:11311'
#os.environ['ROS_IP'] = '10.43.47.222'
#os.environ['ROS_IP'] = '10.43.46.100'
#os.environ['ROS_IP'] = '10.43.47.30'



#rospy.init_node("hoge")
#rospy.loginfo('start')
#sub = rospy.Subscriber("sub", std_msgs.msg.String, callback)
#pub = rospy.Publisher('pub', std_msgs.msg.Int16, queue_size=10)
#pose_d = rospy.Publisher('camera_frame', Pose, queue_size=10)
#pose_d_msg = Pose()
#pub = rospy.Publisher('pose_array_camera', PoseArray, queue_size=10)

#rate = rospy.Rate(4)



weights = "best640lobb.engine"
conf_thres = 0.1
iou_thres = 0.7
improvement_thresh = -.7  #Ayuda con la sobre o infra exposicion   - 1-1  recomendado  0 - 0.6
ims = 640
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared,640,480,rs.format.y8,30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 0)
    #filling = rs.hole_filling_filter(1)
    
align_to = rs.stream.color
align_color = rs.align(align_to)
#align_to = rs.stream.depth
#align_depth = rs.align(align_to)

model = YOLO(weights,task='obb')  # pretrained YOLOv8n model
model.names[0]  = "U"
model.names[1]  = "R"
names =  model.names

#########################################################
stream = profile.get_stream(rs.stream.color) # Downcast to video_stream_profile and fetch intrinsics
intr = stream.as_video_stream_profile().get_intrinsics()
print ("fx: "    + str(intr.fx))
print ("fy: "    + str(intr.fy))
print ("ppx: "   + str(intr.ppx))
print ("ppy: "   + str(intr.ppy))
print("width: "  + str(intr.width))
print("height: " + str(intr.height))
print("coeffs: " + str(intr.coeffs))
print("model: " + str(intr.model))

###########################################################





while not rospy.is_shutdown():
    poses_array = PoseArray()
    poses_array.header = Header(frame_id='base_link')  # Replace 'base_link' with your desired frame ID
    contador = 0
    frames = pipeline.wait_for_frames()
    frames = align_color.process(frames)
    #frames_color =   filling.process(frames).as_frameset()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        print("No frames received")
        continue
    img =  _improveLight(color_frame.get_data(),improvement_thresh)
    pred = model.detect(img,verbose=False,conf=conf_thres,iou = iou_thres,imgsz=640)
    results = pred[0]
    annotated_frame = pred[0].plot()
    if model.task == 'obb':
        iterate = results.obb
    else:
        iterate = results.boxes
    conttador = 0
    fruits   =  []
    for box in iterate:
        contador = contador +1
        if model.task == 'obb':
            inftensor = box.xywhr.cpu().data.numpy()
            u = round(inftensor[0][0])
            v = round(inftensor[0][1])
            #z = depth_frame(u,v)
            distance = getmeanDistance(11,depth_frame,u,v)
            print("class = "+results.names[int(box.cls)] +" coordinates=" + str(box.xywhr.cpu().data.numpy()) +"distance: " + str(distance)+ "\n")
        else:
            inftensor = box.xywh.cpu().data.numpy()
            u = round(inftensor[0][0])
            v = round(inftensor[0][1])
            #z = depth_frame(u,v)
            distance = getmeanDistance(7,depth_frame,u,v)
            print("class = "+results.names[int(box.cls)] +" coordinates=" + str(box.xywh.cpu().data.numpy())+"distance: " + str(distance)+ "\n")
        if results.names[int(box.cls)] == "R" and distance > 0:
            position =  rs.rs2_deproject_pixel_to_point(intr, [u, v], distance)
            fruit = fruitPos(position[0], position[1], position[2], results.names[int(box.cls)]) 
            print("The diference is: "+ str(fruit.z - distance))
            #fruit = fruitPos( position[2], -position[0], -position[1],results.names[int(box.cls)]) 
            fruits.append(fruit)
        
        #
        #print("x: " +  str(x) + " y: " + str(y) + " z: " + str(z)+ "\n")
    #rospy.sleep()
    time.sleep(1)
    for fruit in fruits:
        pose = Pose()
        pose.position.x = fruit.x
        pose.position.y = fruit.y
        pose.position.z = fruit.z
        poses_array.poses.append(pose)
    
    cv2.imshow("Inference", annotated_frame)
    if poses_array:
 #       pose_d.publish(pose_d_msg)
  #      pub.publish(poses_array)
     
    #if contador > 0:
    #    pose_d_msg.position.x = x #round(inftensor[0][0])
    #    pose_d_msg.position.y = y #round(inftensor[0][1])
    #    pose_d_msg.position.z = z #distance
    #    pose_d_msg.orientation.x = 0
    #    pose_d_msg.orientation.y = 0
    #    pose_d_msg.orientation.x = 0
    #    pose_d_msg.orientation.w = 1
        key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
