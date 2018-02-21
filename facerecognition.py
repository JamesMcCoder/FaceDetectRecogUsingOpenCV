import cv2
import numpy as np
import os

fname = "trainner/trainner.yml"
if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)

faceCascade = cv2.CascadeClassifier('/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(fname)

#while True:
#ret, im =cam.read()
im = cv2.imread("test.jpeg",1)
gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray, 1.3, 5)
for(x,y,w,h) in faces:
     cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
     Id,conf = recognizer.predict(gray[y:y+h,x:x+w])
     print (conf)
     print (Id)
     #if conf < 50:
     if(Id==1):
          Id="Angelina"
     elif(Id==2):
          Id="Jennifer"
     elif(Id==3):
          Id="Salma"
     elif(Id==4):
          Id="Sharon"
     else:
          Id="Unknown"
     print (Id)
     newimg = cv2.putText(im, str(Id), (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
    #cv2.imshow('im',im) 
     cv2.imwrite('/var/www/html/iot/result.jpg',newimg)
    #if cv2.waitKey(10) & 0xFF == ord('q'):
    #    break

cam.release()
cv2.destroyAllWindows()