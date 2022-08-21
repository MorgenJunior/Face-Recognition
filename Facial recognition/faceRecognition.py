
from turtle import Vec2D
import cv2
import random as randrange

trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choosing what tp read
# img = cv2.imread('jeff_bezos2.jpg')
img = cv2.imread('jeff_bezos2.jpg')



#converting image to GrayScale
grayColoredImg = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#Detect Face
face_coordinates = trained_face_data.detectMultiScale(grayColoredImg)

#draw rectangles around the face
# cv2.rectangle(img, (57 ,34),( 57+102, 34+102), (0,225,0),2)
# cv2.rectangle(img, (x , y),( x+w, y+h), (0,225,0),2)
for (x, y, w ,h) in  face_coordinates:
    cv2.rectangle(img, (x , y),( x+w, y+h), (0, 255, 0),2)




#showing whats been read
cv2.imshow("Ppencv Feed", img)
cv2.waitKey()


print ("complete")