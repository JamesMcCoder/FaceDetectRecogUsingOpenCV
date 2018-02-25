
import cv2
import numpy as np
import os

fname = "trainner/trainner.yml"
path = 'testSet'
increment = 0

if not os.path.isfile(fname):
    print("Please train the data first")
    exit(0)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFaceRecognizer_create()
#recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read(fname)

#get the path of all the files in the folder
imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

#now looping through all the image paths and loading the Ids and the images
for imagePath in imagePaths:
    increment = increment + 1
    im = cv2.imread(imagePath,1)
    origFileName=os.path.split(imagePath)[-1].split('.')[0]
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
         cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
         
         thisImg = gray[y:y+h,x:x+w]
         r = 150.0 / thisImg.shape[1]
         dim = (150, int(thisImg.shape[0] * r))
         # perform the actual resizing of the image and show it
         resized = cv2.resize(thisImg, dim, interpolation = cv2.INTER_AREA)
            
         Id,conf = recognizer.predict(resized)
         print (conf)
         #if conf < 50:
         if(Id==1):
              Id="Angelina"
         elif(Id==2):
              Id="Jennifer"
         elif(Id==3):
              Id="Selma"
         elif(Id==4):
              Id="Sharon"
         else:
              Id="Unknown"
         newimg = cv2.putText(im, str(Id), (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
         newName = origFileName+ "-" + Id + str(increment)
         print (newName)
         print ("------")
         cv2.imwrite('/var/www/html/iot/results/'+ newName +'.jpg',newimg)

cam.release()
cv2.destroyAllWindows()
