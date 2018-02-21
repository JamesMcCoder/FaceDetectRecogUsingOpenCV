import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
orig = 'origSet'
path = 'dataSet'
face_cascade = cv2.CascadeClassifier('/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_default.xml')

#get the path of all the files in the folder
imagePaths=[os.path.join(orig,f) for f in os.listdir(orig)] 

#now looping through all the images and greyscale them and cut out faces
for imagePath in imagePaths:
    origFileName=os.path.split(imagePath)[-1]
    #loading the image and converting it to gray scale
    #pilImage=Image.open(imagePath).convert('L')
    # Load an color image in grayscale
    img = cv2.imread(imagePath,0)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.imwrite("dataSet/"+origFileName,img[y:y+h,x:x+w])
        
cv2.destroyAllWindows()