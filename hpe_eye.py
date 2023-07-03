# import cv2
# import numpy as np
# # import dlib
# #  https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
# # Read Image
# # im = cv2.imread("./examples/example1.jpg")
# im = cv2.imread("headPose.jpg")
# size = im.shape
 
# #2D image points. If you change the image, you need to change vector
# image_points = np.array([
#                             (359, 391),     # Nose tip
#                             (399, 561),     # Chin
#                             (337, 297),     # Left eye left corner
#                             (513, 301),     # Right eye right corne
#                             (345, 465),     # Left Mouth corner
#                             (453, 469)      # Right mouth corner
#                         ], dtype="double")
 
# # 3D model points.
# model_points = np.array([
#                             (0.0, 0.0, 0.0),             # Nose tip
#                             (0.0, -330.0, -65.0),        # Chin
#                             (-225.0, 170.0, -135.0),     # Left eye left corner
#                             (225.0, 170.0, -135.0),      # Right eye right corne
#                             (-150.0, -150.0, -125.0),    # Left Mouth corner
#                             (150.0, -150.0, -125.0)      # Right mouth corner
 
#                         ])
            
# # Camera internals
 
# focal_length = size[1]
# center = (size[1]/2, size[0]/2)
# camera_matrix = np.array(
#                          [[focal_length, 0, center[0]],
#                          [0, focal_length, center[1]],
#                          [0, 0, 1]], dtype = "double"
#                          )
 
# print ("Camera Matrix :\n {0}".format(camera_matrix))
 
# dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
# (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
 
# print ("Rotation Vector:\n {0}".format(rotation_vector))
# print ("Translation Vector:\n {0}".format(translation_vector))
 
# # Project a 3D point (0, 0, 1000.0) onto the image plane.
# # We use this to draw a line sticking out of the nose
 
# (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
 
# for p in image_points:
#     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
 
# p1 = ( int(image_points[0][0]), int(image_points[0][1]))
# p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

# cv2.line(im, p1, p2, (255,0,0), 2)
 
# # Display image
# cv2.imshow("Output", im)
# cv2.waitKey(0)


import numpy as np
import cv2
import autopy
from matplotlib import pyplot as plt
import pygame
# https://github.com/Doruk-Dilmen/Python-Gaze-estimation--Eye-tracking--using-single-low-cost-web-am-and-visualization-of-data/blob/main/Gaze_Estimation.py
# https://www.youtube.com/watch?v=1UihotbMCkY&ab_channel=DorukDilmen

ESCAPE_KEY = 27

k=1
sumx=0
sumy=0
orn=7

cap = cv2.VideoCapture(0)

screen_resolution=(640,480)
video_resolution=(640,480)


pygame.init()
screen =pygame.display.set_mode((640,480))

screen_resolution = autopy.screen.size()

eye_x_positions = list()
eye_y_positions = list()

while 1:
    success, image = cap.read()
    image=cv2.flip(image,1)
    roi = image[150:250 , 230:330]
    resized1 = cv2.resize(roi, (200,200), interpolation = cv2.INTER_AREA)
    
    cv2.circle(resized1, (160, 127), 2, (155, 155, 255), 4)
    cv2.circle(resized1, (50, 127), 2, (155, 155, 255), 4)
    
    cv2.imshow("Parça",resized1)
    
    resized1=resized1[80:170 , 50:160] 
    resized = cv2.resize(resized1, (440,360), interpolation = cv2.INTER_AREA)
    rows,cols,_ = resized1.shape
    gray1 = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    eye_blur = cv2.bilateralFilter(gray1,  10, 195,195)
    cv2.imshow("eye_blur",eye_blur)
    img_blur = cv2.Canny(eye_blur,10,30)
    
    cv2.imshow("img_blur",img_blur)
    #img_blur = cv2.Canny(eye_blur,10,51)
    #eye_blur = cv2.bilateralFilter(gray1,  10, 80,95)
    #img_blur = cv2.Canny(eye_blur,10,35)   
    
    
    cv2.imshow('Canny', img_blur)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 0.1, 400, param1=200, param2=10, minRadius=76, maxRadius=84)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(resized, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(resized, (i[0], i[1]), 2, (0, 0, 255), 5)
            print(i[0],i[1])
            #pygame.draw.circle(screen, (0,0,255), ((i[0]-64)*17, (i[1]-60)*40), 5)
            #cv2.putText(frame,"Left eye x location = " + str(i[0]) , (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (155, 255, 0), 2)
            #cv2.putText(frame,"Left eye y location = " + str(i[1]) , (20,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (155, 255, 0), 2)
            if k==orn:
                k=1
                sumx=sumx/orn
                sumx=round(sumx,2)
                sumy=sumy/orn
                sumy=round(sumy,2)
                print("_______\n",sumx,sumy,"\n_______")
                eye_x_p=round((sumx-145),2)
                #eye_y_p=(sumy-62)
                eye_y_p=round((sumy-145),2)
                pygame.draw.circle(screen, (0,0,255), (eye_x_p*4, eye_y_p*9), 5)
                
               
                
                eye_x_positions.append(eye_x_p)
                eye_y_positions.append(eye_y_p)
                
                
            elif k==0:
                pygame.display.update()
                screen.fill((0,0,0))
                sumx=0
                sumy=0
                k=k+1
                
            elif k==1:
                pygame.display.update()
                screen.fill((0,0,0))
                sumx=sumx+i[0]
                sumy=sumy+i[1]
                k=k+1
                
            else:
                sumx=sumx+i[0]
                sumy=sumy+i[1]
                k=k+1
            
            cv2.putText(image, str(i[0]) , (200,30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (155, 255, 0), 2)
            cv2.putText(image, str(i[1]) , (200,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (155, 255, 0), 2)
    
    
    cv2.putText(image,"Left eye x location = " , (20,30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (155, 255, 0), 2)
    cv2.putText(image,"Left eye y location = " , (20,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (155, 255, 0), 2)
    cv2.imshow("Eye", resized)
    #cv2.imshow("Roi", resized1)
    
    cv2.imshow("frame", image) 
    key_pressed = cv2.waitKey(1) & 0xff
    if key_pressed == ESCAPE_KEY:
        break


data_all = list(zip(eye_x_positions,eye_y_positions ))
print(data_all)
plt.scatter(eye_x_positions, eye_y_positions ,color="blue")
plt.title("Eye position")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.axis([0, 150, 55, 0])
plt.show()


'''
x_axis_labels = [0,10,20,30,40,50,60,70,80,90,100] 
y_axis_labels = [0,2.5,5,7.5,10,12.5,15,17.5,20]
sns.heatmap(data_all, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cbar=False)
plt.xlabel("X position")
plt.ylabel("Y position")
plt.show()

vid.release()

'''

pygame.display.quit()
cap.release()
cv2.destroyAllWindows()