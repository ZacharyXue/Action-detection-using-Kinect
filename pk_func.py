from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
import cv2

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
