
import numpy as np
from skimage.color import rgb2gray
from skimage import io
import math
import cv2
import matplotlib.pyplot as plt 
from PIL import Image, ImageEnhance

def estimate(I,darkChannel,p):
    M, N = darkChannel.shape
    flatI = I.reshape(M*N, 3)
    flatbright = darkChannel.ravel() # make array flatten
    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # find top M * N * p indexes. argsort() returns sorted (ascending) index.
    # return the mean intensity for each channel
    A = np.mean(flatI.take(searchidx, axis=0),dtype=np.float64, axis=0) # 'take' get value from index.
    return A

def _improveLight(img,omega):
    if omega < 0:
        img = np.asarray(img)
        img = Image.fromarray(img).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        F  = enhancer.enhance(abs (omega))
    else:
        img = np.asarray(img)
        A = img.astype('float64')/255
        B = 1 - img.astype('float64')/255
        darkChannel = np.min(B,   axis= 2)
        atmLight = estimate(B,darkChannel,0.1)
        normI = B/atmLight
        normI = np.min(B,axis = 2)
        T =  np.stack((normI,)*3, axis=-1)
        transmissionMap = 1 - omega *  normI
        radianceMap = atmLight + np.divide((B-atmLight) , np.stack((transmissionMap,)*3, axis=-1))
        radianceMap = 1 - np.minimum (1,np.maximum(0,radianceMap))
        F = A*(1-T) + radianceMap*(T)
        F = (F*255).astype(np.uint8)
    return np.asanyarray(F)

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return int(qx), int(qy)



def _xml_to_yolo_bbox_normalized(bbox,m,n,degrees):
    # xmin, ymin, xmax, ymax
    rectangle_rotated = np.zeros((0, 4), dtype=np.double)
    x = ((bbox[2] + bbox[0]) / 2)
    y = ((bbox[3] + bbox[1]) / 2)
    w = (bbox[2] - bbox[0]) 
    h = (bbox[3] - bbox[1]) 
    pt1 = [(x - w/2), (y - h/2)]
    pt2 = [(x - w/2), (y + h/2)]
    pt3 = [(x + w/2), (y +  h/2)]
    pt4 = [(x + w/2), (y - h/2)]
    center = [x, y]
    pt1 = rotate(center,pt1,math.radians(degrees))
    pt2 = rotate(center,pt2,math.radians(degrees))
    pt3 = rotate(center,pt3,math.radians(degrees))
    pt4 = rotate(center,pt4,math.radians(degrees))
    

  

    pt1 = [max(0,min(1, pt1[0]/m)), max(0,min(1,pt1[1]/n))]
    pt2 = [max(0,min(1,pt2[0]/m)), max(0,min(1,pt2[1]/n))]
    pt3 = [max(0,min(1,pt3[0]/m)), max(0,min(1,pt3[1]/n))]
    pt4 = [max(0,min(1,pt4[0]/m)), max(0,min(1,pt4[1]/n))]
    rectangle_rotated = np.concatenate([pt1, pt2, pt3, pt4],axis=-1)
    return rectangle_rotated

def _xml_to_yolo_bbox(bbox,degrees):
    # xmin, ymin, xmax, ymax
    rectangle_rotated = np.zeros((0, 4), dtype=np.double)
    x = ((bbox[2] + bbox[0]) / 2)
    y = ((bbox[3] + bbox[1]) / 2)
    w = (bbox[2] - bbox[0]) 
    h = (bbox[3] - bbox[1]) 
    pt1 = [(x - w/2), (y - h/2)]
    pt2 = [(x - w/2), (y + h/2)]
    pt3 = [(x + w/2), (y +  h/2)]
    pt4 = [(x + w/2), (y - h/2)]
    center = [x, y]
    pt1 = rotate(center,pt1,math.radians(degrees))
    pt2 = rotate(center,pt2,math.radians(degrees))
    pt3 = rotate(center,pt3,math.radians(degrees))
    pt4 = rotate(center,pt4,math.radians(degrees))
    pt1 = [pt1[0], pt1[1]]
    pt2 = [pt2[0], pt2[1]]
    pt3 = [pt3[0], pt3[1]]
    pt4 = [pt4[0], pt4[1]]
    rectangle_rotated = np.concatenate([pt1, pt2, pt3, pt4],axis=-1)
    return rectangle_rotated

def yolo_to_xml_bbox(bbox, w, h):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]


def _invertBbox(bbox,w):
    for i in [0,2,4,6]:
        bbox[i] = w-bbox[i]
    pt1 = bbox[[2,3]]
    pt2 = bbox[[0,1]]
    pt3 = bbox[[6,7]]
    pt4 = bbox[[4,5]]
    return np.concatenate([pt1, pt2, pt3, pt4],axis=-1)



def _print(img,bboxes):
    for box in bboxes:
        x1 = box[[0,1]]
        x2 = box[[2,3]]
        x3 = box[[4,5]]
        x4 = box[[6,7]]
        pts = np.array([x1,x2,x3,x4],np.int32)
        pts = pts.reshape((-1,1,2))
        #print(pts)
        img = cv2.polylines(np.array(img),[pts],True,(0,255,255))
    c = plt.imshow(img)
    plt.show()




def _xml_to_yolo_NOoriented_bbox(bbox,m,n,degrees):
    # xmin, ymin, xmax, ymax
    factor = .25
    d = factor *( degrees%90/45 if (degrees%90 <= 45) else (abs(degrees%90-90)/45))
    #print(degrees)
    #print(d*factor)
    rectangle_rotated = np.zeros((0, 4), dtype=np.int32)
    x = ((bbox[2] + bbox[0]) / 2)
    y = ((bbox[3] + bbox[1]) / 2)
    w = (bbox[2] - bbox[0])
    h = (bbox[3] - bbox[1])
    w = w + w*d
    h = h + h*d 
    pt1 = [x - w/2, y - h/2]
    pt2 = [x - w/2, y + h/2]
    pt3 = [x + w/2, y +  h/2]
    pt4 = [x + w/2, y - h/2]
    center = [x, y]
    pt1 = rotate(center,pt1,math.radians(degrees))
    pt2 = rotate(center,pt2,math.radians(degrees))
    pt3 = rotate(center,pt3,math.radians(degrees))
    pt4 = rotate(center,pt4,math.radians(degrees))
    
    xx = [pt1[0], pt2[0], pt3[0], pt4[0]]
    yy = [pt1[1], pt2[1], pt3[1], pt4[1]]


    xx = [(xx[0] + xx[3])/2 , (xx[0] + xx[1])/2, (xx[1] + xx[2])/2, (xx[2] + xx[3])/2]
    yy = [(yy[0] + yy[3])/2 , (yy[0] + yy[1])/2, (yy[1] + yy[2])/2, (yy[2] + yy[3])/2]
   
    rectangle_rotated = np.concatenate([ [min(xx),min(yy)],[min(xx),max(yy)],[max(xx),max(yy)],[max(xx),min(yy)]],axis=-1)
    
    
    return rectangle_rotated

def _xml_to_yolo_NOoriented_bbox_normalized(bbox,m,n,degrees):
    factor = .25
    d = factor *( degrees%90/45 if (degrees%90 <= 45) else (abs(degrees%90-90)/45))
    # xmin, ymin, xmax, ymax
    rectangle_rotated = np.zeros((0, 4), dtype=np.int32)
    x = ((bbox[2] + bbox[0]) / 2)
    y = ((bbox[3] + bbox[1]) / 2)
    w = (bbox[2] - bbox[0])
    h = (bbox[3] - bbox[1])
    w = w + w*d
    h = h + h*d 
    pt1 = [x - w/2, y - h/2]
    pt2 = [x - w/2, y + h/2]
    pt3 = [x + w/2, y +  h/2]
    pt4 = [x + w/2, y - h/2]
    center = [x, y]
    pt1 = rotate(center,pt1,math.radians(degrees))
    pt2 = rotate(center,pt2,math.radians(degrees))
    pt3 = rotate(center,pt3,math.radians(degrees))
    pt4 = rotate(center,pt4,math.radians(degrees))
    
    xx = [pt1[0]/m, pt2[0]/m, pt3[0]/m, pt4[0]/m]
    yy = [pt1[1]/n, pt2[1]/n, pt3[1]/n, pt4[1]/n]

    xx = [(xx[0] + xx[3])/2 , (xx[0] + xx[1])/2, (xx[1] + xx[2])/2, (xx[2] + xx[3])/2]
    yy = [(yy[0] + yy[3])/2 , (yy[0] + yy[1])/2, (yy[1] + yy[2])/2, (yy[2] + yy[3])/2]
   
    rectangle_rotated = np.concatenate([ [min(xx),min(yy)],[max(xx),min(yy)],[min(xx),max(yy)],[max(xx),max(yy)]],axis=-1)
    return rectangle_rotated