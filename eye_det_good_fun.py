# code from: https://github.com/JasonCaoCJX/Eye-closure-detection-EAR 
# paper分析&code : https://blog.csdn.net/hongbin_xu/article/details/79033116
# paper: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf?spm=5176.100239.blogcont336184.8.b7697a07zKT7r&file=05.pdf
from scipy.spatial import distance as dist  # good module to calculate distance
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
# import dlib
import cv2


def eye_ratio(eye):
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)  # 人眼纵横比(eye aspect ratio)
    return ear


def eye_detection(frame,threshold, sustain):
    # 人眼纵横比参数
    EAR_THRESHOLD = threshold  # 阈值
    SUSTAIN_FREAMS = sustain  # 当检测到人眼超过50帧还在闭眼状态，说明人正在瞌睡

    # 检测帧次数
    COUNTER = 0

    # 加载人脸68点数据模型
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # model

    # 获取人眼的坐标
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    # 从摄像头中来获取人脸
    # vs = VideoStream(src=0).start()
    time.sleep(1.0)

    # 从视频中获取视频帧来进行检测
    # while True:
        
        # 从视频中获取图片来检测
        # frame = vs.read()
    frame = imutils.resize(frame, width=700)
    # cv2.imshow("original_image" , frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 提取人眼坐标，来计算人眼纵横比
        leftEye = shape[lStart:lEnd]  # 43點~46點  公式:P2-P6
        rightEye = shape[rStart:rEnd]  # 37點~40點 公式:P3-P5
        leftEAR = eye_ratio(leftEye)
        rightEAR = eye_ratio(rightEye)

        # 平均左右眼的纵横比
        ear = (leftEAR + rightEAR) / 2.0  # Eye Aspect Ratio

        # Show 左右眼 畫左右眼眼睛的外圍
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)


        # 计算ear是否小于阈值
        if ear < EAR_THRESHOLD:  # EAR_THRESHOLD = threshold 0.23



            COUNTER += 1
            if COUNTER >= SUSTAIN_FREAMS:
                # 人瞌睡是要处理的函数
                cv2.putText(frame, "You Died", (230, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)

            # 闭眼时，提醒
            if COUNTER >= 10 :
                cv2.putText(frame, 'WARNING {:.2f}'.format(ear), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)



    # cv2.imshow("EAR_Frame", frame)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break

    # cv2.destroyAllWindows()
    # vs.stop()










