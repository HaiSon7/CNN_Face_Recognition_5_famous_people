import cv2
import numpy as np
import os

face_detection = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")

cam = cv2.VideoCapture(0)
i = 0
while True:
    OK , frame = cam.read()
    faces = face_detection.detectMultiScale(frame,1.38,5)
    for (x,y,w,h) in faces:
        roi = cv2.resize(frame[y+2:y+h -2 ,x+2:x+w-2],(100,100))

        cv2.imwrite('image/img_faces/Train/HaiSon/img_{}.jpg'.format(i),roi)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        i+=1

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
