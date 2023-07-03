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
from hpe import HPEstimator
estimator = HPEstimator()
from math import cos,sin
import winsound
# from hpe_axis import Bilinear_interpolation

# code: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py

mp_face_mesh = mp.solutions.face_mesh
import mediapipe.python.solutions.face_mesh_connections as face_mesh_connections

# 用 face_mesh 來估測後續的歐拉角  ***  初始化Pose Estimation模型
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
# pitch,yaw,pitch_LT,yaw_LT,pitch_RT,yaw_RT,pitch_RB,yaw_RB,pitch_LB,yaw_LB = 0

# 加载人脸68点数据模型
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # model
path = "Calibration.txt"

# # 获取人眼的坐标
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

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
        for face_landmarks in results.multi_face_landmarks:
            #  468+10 landmarks
            mp_drawing.draw_landmarks(facemesh_fig,face_landmarks,face_mesh_connections.FACEMESH_CONTOURS,mp_drawing.DrawingSpec((0,255,0),1,1))  # draw landmark
            
            for idx, lm in enumerate(face_landmarks.landmark):  # enumerate function get face idx number and value  https://clay-atlas.com/blog/2019/11/08/python-chinese-function-enumerate/
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:  # idx=1 is nose
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    print("idx,lm, x,y" , idx,lm, x,y)
                    
                    # Get the 2D Coordinates
                    face_2d.append([x, y])
                
                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       

            # print("results.multi_face_landmarks" , results.multi_face_landmarks)
            
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
            print("dist_matrix",dist_matrix)

            # Solve PnP ***  success detectpoint on face?  rot_vec, trans_vec : R T
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            # print("rot_vec :",rot_vec )

            # Get rotational matrix  rotation vector to matrix
            rmat, jac = cv2.Rodrigues(rot_vec) 

            # Get three Euler angles of rotation in degrees.
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # decomposeProjectionMatrix()
            # print("Qx, Qy, Qz",Qx, Qy, Qz)  # 迴繞在x,y,z的旋轉矩陣
            # Get the y rotation degree
            x = angles[0] * 360  
            y = angles[1] * 360  
            z = angles[2] * 360

            # (pitch , yaw , roll) = (np.round(x,2),np.round(y,2),np.round(z,2)) 

            # 角度轉為弧度  https://www.shuxuele.com/geometry/radians.html
            x180 = x  *np.pi/180  # pitch
            y180 = y  *np.pi/180  # yaw
            z180 = z  *np.pi/180  # roll
        ### ----------- Tai -------------- ###
            # yaw, pitch, roll , bboxs  = estimator.predict(ori_image,visualize=False)
            # hpe_pose,roll,pitch,yaw = estimator.draw(ori_image,bboxs,yaw, pitch, roll)
            # cv2.imshow("hpe_pose",hpe_pose)
            cv2.putText(image, "roll: " + str(np.round(z,2)), (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "pitch: " + str(np.round(x,2)), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "yaw: " + str(np.round(y,2)), (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "shott_num: " + str(shott_num), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    ### ------------ calibration ------------ ###
        # if 270 <= nose_2d[0] <= 390:
        #     if 130 <= nose_2d[1] <= 350:
            ### ----------- Tai -------------- ###
            tdx = (int (nose_2d[0] ))
            tdy = (int (nose_2d[1] ))
            size = image.shape[0] # row
             
            # use Tait–Bryan angles X1 Y2 Z3

            x3 = size * (cos(y180) * cos(z180)) +  tdx # x-axis
            y3 = size * (cos(x180) * sin(z180) + cos(z180) * sin(x180) * sin(y180)) + tdy  # x-axis
            x2 = size * (-cos(y180) * sin(z180)) + tdx  # z-axis
            y2 = size * (cos(x180) * cos(z180) - sin(x180) * sin(y180) * sin(z180)) + tdy  # z-axis
            x1 = size * (sin(y180)) + tdx  
            y1 = size * (-cos(y180) * sin(x180)) + tdy  
        ### ----------- Tai -------------- ###
            # ori_image = cv2.circle(ori_image,(mouse_x,mouse_y),10,(255,0,0),-1)
           
            # Display the nose direction (reprohection point)
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
        
            p1 = (int(nose_2d[0]),int(nose_2d[1]))  # nose position
            p2 = (int(nose_2d[0] + y * 5) , int(nose_2d[1] - x * 5))
            # print("p1",p1)
            # cv2.line(image, p1, p2, (255, 0, 0), 3)
            # image = cv2.line(image,(round(tdx), round(tdy)), (round(x3), round(y3)),(0,0,255),5)  # y-axis pitch
            # image = cv2.line(image,(round(tdx), round(tdy)), (round(x2), round(y2)),(0,255,0),5)  # z-axis yaw
            # image = cv2.line(image,(round(tdx), round(tdy)), (round(x1), round(y1)),(255,0,0),5)  # x-axis roll
            ori_image = cv2.circle(ori_image,(round(tdx),round(tdy)),5,(255,0,0),-1) 
            ori_image = cv2.circle(ori_image,(round(x1),round(y1)),8,(0,0,255),-1) 
            ED = math.sqrt(math.pow(round(tdx-x1),2)+math.pow(round(tdy-y1),2)) 
            cv2.putText(image, "ED: " + str(round(ED)), (450, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "x: " + str(round(abs(tdx-x1))), (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(round(abs(tdy-y1))), (450, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        # print("data_save",data_save)
        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        
        #   不顯示臉mesh
        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # FACE_CONNECTIONS
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    cv2.imshow('Demo', ori_image)
    cv2.imshow('Head Pose Estimation', image)
    cv2.imshow('facemesh', facemesh_fig)

    # cv2.namedWindow('Full_screen', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('Full_screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.imshow('Full_screen', ori_image)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break


cap.release()