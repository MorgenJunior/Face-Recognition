
from turtle import Vec2D
import cv2
import random as randrange

trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#choosing what tp read, 0 will open the web cam
webcam = cv2.VideoCapture(0)

while True:
    sucessflu_frame_read, frame = webcam.read()

    #converting image to GrayScale
    grayColoredImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # #Detect Face
    face_coordinates = trained_face_data.detectMultiScale(grayColoredImg)

    
    # #draw rectangles around the face
    for (x, y, w ,h) in  face_coordinates:
         cv2.rectangle(frame, (x , y),( x+w, y+h), (0, 255, 0),2)



    #showing whats been read
    cv2.imshow("OPencv Feed", frame)
    key = cv2.waitKey(1)

    #stoping the webcam(Q or q)
    if key == 81 or key ==113:
        break

webcam.release()




print ("complete")