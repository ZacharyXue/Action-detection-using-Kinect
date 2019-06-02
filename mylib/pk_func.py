from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
import cv2

import ctypes

import math

import numpy as np

def draw_body_bone(joints, jointPoints, joint0, joint1,img):
    joint0State = joints[joint0].TrackingState
    joint1State = joints[joint1].TrackingState

    # both joints are not tracked
    if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
        return

    # both joints are not *really* tracked
    if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
        return

    start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
    end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))
    cv2.line(img, start,end,(0, 0, 255),8)

def draw_body(joints, jointPoints,img):
    # Torso
    draw_body_bone(joints, jointPoints ,PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck,img)
    draw_body_bone(joints, jointPoints, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase,img)
    draw_body_bone(joints, jointPoints, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight,img)
    draw_body_bone(joints, jointPoints, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft,img)

    # Right Arm    
    draw_body_bone(joints, jointPoints, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight,img)
    draw_body_bone(joints, jointPoints, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight,img)
    draw_body_bone(joints, jointPoints,PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight,img)
    draw_body_bone(joints, jointPoints, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight,img)

    # Left Arm
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft,img)

    # Right Leg
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight,img)

    # Left Leg
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft,img)
    draw_body_bone(joints, jointPoints,  PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft,img)

    return img

def get_world_pos(kinect,skeletons):
    width, height = kinect.color_frame_desc.width, kinect.color_frame_desc.height
    cameraPoint_capacity = ctypes.c_uint(width * height)
    cameraPoint_data_type = PyKinectV2._CameraSpacePoint * cameraPoint_capacity.value

    cameraPointCount = ctypes.POINTER(PyKinectV2._CameraSpacePoint)
    cameraPointCount = ctypes.cast(cameraPoint_data_type(), ctypes.POINTER(PyKinectV2._CameraSpacePoint))
    kinect._mapper.MapColorFrameToCameraSpace(kinect._depth_frame_data_capacity,kinect._depth_frame_data,\
        cameraPoint_capacity,cameraPointCount)

    pos = []
    temp_pos = []
    for skeleton in skeletons:
        k = int(len(skeleton) / 2)
        for i in range(k):
            x = skeleton[2*i]
            y = skeleton[2 * i + 1]
            if math.isinf(x):
                x = 0
            if math.isinf(y):
                y = 0
            temp_pos.append(cameraPointCount[int(x * width + y)].x)
            temp_pos.append(cameraPointCount[int(x * width + y)].y)
            temp_pos.append(cameraPointCount[int(x * width + y)].z)
        pos.append(temp_pos)
        temp_pos = []
    
    return pos
    
def to_kinect(kinect,skeletons):
    # get position
    pos = get_world_pos(kinect,skeletons)
    try:
        temp_skeleton = np.array(pos[0])      
    except:
        skeleton = np.array([])
    else:
        # create output
        skeleton = np.random.rand(1,75)
        # print(temp_skeleton.shape)
        # 0->2
        skeleton[0,2*3:3*3] = temp_skeleton[0*3:1*3]
        # 1->20
        skeleton[0,20*3:21*3] = temp_skeleton[1*3:2*3]
        # 2->4
        skeleton[0,4*3:5*3] = temp_skeleton[2*3:3*3]
        # 3->5
        skeleton[0,5*3:6*3] = temp_skeleton[3*3:4*3]
        # 4->6
        skeleton[0,6*3:7*3] = temp_skeleton[4*3:5*3]
        # 5->8
        skeleton[0,8*3:9*3] = temp_skeleton[5*3:6*3]
        # 6->9
        skeleton[0,9*3:10*3] = temp_skeleton[6*3:7*3]
        # 7->10
        skeleton[0,10*3:11*3] = temp_skeleton[7*3:8*3]
        # 8->12
        skeleton[0,12*3:13*3] = temp_skeleton[8*3:9*3]
        # 9->13
        skeleton[0,13*3:14*3] = temp_skeleton[9*3:10*3]
        # 10->14
        skeleton[0,14*3:15*3] = temp_skeleton[10*3:11*3]
        # 11->16
        skeleton[0,16*3:17*3] = temp_skeleton[11*3:12*3]
        # 12->17
        skeleton[0,17*3:18*3] = temp_skeleton[12*3:13*3]
        # 13->18
        skeleton[0,18*3:19*3] = temp_skeleton[13*3:14*3]
        

    return skeleton