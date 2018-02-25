import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
orig = 'origSet'
path = 'dataSet'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#get the path of all the files in the folder
imagePaths=[os.path.join(orig,f) for f in os.listdir(orig)] 

#now looping through all the images and greyscale them and cut out faces
for imagePath in imagePaths:
    origFileName=os.path.split(imagePath)[-1]
    # Load an color image in grayscale
    img = cv2.imread(imagePath,0)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        imgrey = img[y:y+h,x:x+w]
        #wanting size to be 150x150 pixels
        r = 150.0 / imgrey.shape[1]
        dim = (150, int(imgrey.shape[0] * r))
        # perform the actual resizing of the image and show it
        resized = cv2.resize(imgrey, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite("dataSet/"+origFileName,resized)
        
cv2.destroyAllWindows()
