import cv2
import os
import mediapipe as mp
import numpy as np
import time
import math
# import dlib  # ***
from scipy.spatial import distance as dist  # good module to calculate distance
from imutils.video import VideoStream
from imutils import face_utils
import imutils
from eye_det_good_fun import eye_ratio

from math import cos,sin
# import winsound
# from hpe_axis import Bilinear_interpolation

# code: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py

import mediapipe.python.solutions.face_mesh_connections as face_mesh_connections

mp_face_mesh = mp.solutions.face_mesh

# 初始化 Mediapipe Face Mesh 模組
face_mesh = mp_face_mesh.FaceMesh()

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

EAR_THRESHOLD = 0.2  # 阈值

# 检测帧次数
COUNTER = 0
COUNTER_off = 0
shott_num = 0 # 點擊次數
data_save = 0
game_shott = 0
game_shott_timer = 0

path = "Calibration.txt"

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("output.mp4")

while cap.isOpened():
    success, image = cap.read()
    start = time.time()
    ori_image = cv2.flip(image,1,dst=None) #水平镜像 
    facemesh_fig = cv2.flip(image,1,dst=None) #水平镜像 
    # print("size",ori_image.shape)


    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []


    if results.multi_face_landmarks:
        # ------------ calculate angle ----------------- #
        # 取出第一張臉的特徵點
        landmarks = results.multi_face_landmarks[0]

        # 取出左眼、右眼和嘴巴的位置
        left_eye = landmarks.landmark[33].x, landmarks.landmark[33].y
        right_eye = landmarks.landmark[263].x, landmarks.landmark[263].y
        nose = landmarks.landmark[1].x, landmarks.landmark[1].y
        mouth_left = landmarks.landmark[61].x, landmarks.landmark[61].y
        mouth_right = landmarks.landmark[291].x, landmarks.landmark[291].y
        
        # 計算左眼和右眼之間的距離
        eye_distance = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)

        # 計算鼻子到左眼、右眼、嘴角的距離
        nose_to_left_eye = math.sqrt((nose[0] - left_eye[0])**2 + (nose[1] - left_eye[1])**2)
        nose_to_right_eye = math.sqrt((nose[0] - right_eye[0])**2 + (nose[1] - right_eye[1])**2)
        nose_to_mouth_left = math.sqrt((nose[0] - mouth_left[0])**2 + (nose[1] - mouth_left[1])**2)
        nose_to_mouth_right = math.sqrt((nose[0] - mouth_right[0])**2 + (nose[1] - mouth_right[1])**2)

        # 計算頭部上下轉動角度
        pitch = math.atan2(nose_to_right_eye - nose_to_left_eye, eye_distance)  # y/x
        pitch = math.degrees(pitch)
        # pitch = np.abs(pitch+90)  if pitch < 0 else pitch + 90

        # 計算頭部左右轉動角度
        yaw = math.atan2(nose_to_mouth_right - nose_to_mouth_left, eye_distance)  # y/x
        yaw = math.degrees(yaw)
        # yaw = np.abs(yaw+90)  if yaw < 0 else yaw + 90

        # cv2.putText(image,"Head angle y {}".format(round(pitch,2)), (380, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        # cv2.putText(image,"Head angle x {}".format(round(yaw,2) ), (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # ------------ calculate angle ----------------- #
        for face_landmarks in results.multi_face_landmarks:
            #  468+10 landmarks
            mp_drawing.draw_landmarks(facemesh_fig,face_landmarks,face_mesh_connections.FACEMESH_CONTOURS,mp_drawing.DrawingSpec((0,255,0),1,1))  # draw landmark
            
            for idx, lm in enumerate(face_landmarks.landmark):  # enumerate function get face idx number and value  https://clay-atlas.com/blog/2019/11/08/python-chinese-function-enumerate/
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:  # idx=1 is nose
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # print("idx,lm, x,y" , idx,lm, x,y)
                    
                    # Get the 2D Coordinates
                    face_2d.append([x, y])
                
                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])    


         # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)  # matrix size 4*1
            # print("dist_matrix",dist_matrix)

            # Solve PnP ***  success detectpoint on face?  rot_vec, trans_vec : R T
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            # print("rot_vec :",rot_vec )

            # Get rotational matrix  rotation vector to matrix
            rmat, jac = cv2.Rodrigues(rot_vec) 

            # Get three Euler angles of rotation in degrees.
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # decomposeProjectionMatrix()

            # Get the y rotation degree
            x = angles[0] * 360  
            y = angles[1] * 360  
            z = angles[2] * 360   
            (pitch , yaw , roll) = (np.round(x,2),np.round(y,2),np.round(z,2)) 
            
            cv2.putText(facemesh_fig, "up-down: " + str(np.round(x,2)), (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(facemesh_fig, "left-right: " + str(np.round(y,2)), (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # 角度轉為弧度  https://www.shuxuele.com/geometry/radians.html
            x180 = x  *np.pi/180  # pitch
            y180 = y  *np.pi/180  # yaw
            z180 = z  *np.pi/180  # roll
            # print("x:{},y:{},z:{}".format(x180,y180,z180))

            tdx = (int (nose_2d[0] ))
            tdy = (int (nose_2d[1] ))
            size = image.shape[0] # row

            x3 = size * (cos(y180) * cos(z180)) +  tdx # x-axis
            y3 = size * (cos(x180) * sin(z180) + cos(z180) * sin(x180) * sin(y180)) + tdy  # x-axis
            x2 = size * (-cos(y180) * sin(z180)) + tdx  # z-axis
            y2 = size * (cos(x180) * cos(z180) - sin(x180) * sin(y180) * sin(z180)) + tdy  # z-axis
            x1 = size * (sin(y180)) + tdx  
            y1 = size * (-cos(y180) * sin(x180)) + tdy  

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
            p1 = (int(nose_2d[0]),int(nose_2d[1]))  # nose position
            p2 = (int(nose_2d[0] + y * 5) , int(nose_2d[1] - x * 5))
            ori_image = cv2.line(image,(round(tdx), round(tdy)), (round(x1), round(y1)),(0,0,255),5)  
    

    cv2.imshow('Demo', ori_image)
    cv2.imshow('Head Pose Estimation', image)
    cv2.imshow('facemesh', facemesh_fig)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break


cap.release()