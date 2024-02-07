#Step 1
!pip install opencv-python
import numpy as np
import cv2

#Step 2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Step 3
img = cv2.imread('people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Step 4
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

#Step 5
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

#Step 6
cv2.imshow('img',img)
cv2.waitKey(0)

cv2.destroyAllWindows()

