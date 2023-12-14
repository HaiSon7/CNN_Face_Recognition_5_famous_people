import cv2
import numpy as np
from tensorflow.keras import models


LABELS = ['Bill Gates', 'Elon Musk', 'Jeff Bezos', 'Mark Zuckerberg', 'Steve Jobs']
face_detection = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
cam = cv2.VideoCapture(0)

models = models.load_model("C:\Python\AI\CNN_Algorithm\CNN_Project\Face_detection\model_famous_people_h5")

while True:
    OK , frame = cam.read()
    faces = face_detection.detectMultiScale(frame,1.3,5)
    for (x,y,w,h) in faces:
        roi = cv2.resize(frame[y+2:y+h -2 ,x+2:x+w-2],(70,70))
        p = models.predict(roi.reshape(-1, 70, 70, 3))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(frame,LABELS[np.argmax(p)],(x,y+5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color=(255,255,255),thickness=2)

    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()