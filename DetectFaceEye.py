#PLAN
#1) Detect Frontal Face in video
#2) Detect Both left and right Eyes in video
#3) Count duration of detected eye and face in video

#start

import cv2
import numpy

face_cascade = cv2.CascadeClassifier('/home/luthffi/PycharmProjects/VideoAnalysis/FaceEyeDetection /haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('/home/luthffi/PycharmProjects/VideoAnalysis/FaceEyeDetection /haarcascade_eye.xml')

cap = cv2.VideoCapture('/home/luthffi/PycharmProjects/VideoAnalysis/FaceEyeDetection /03MC_Nabeel.mpg')


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()


