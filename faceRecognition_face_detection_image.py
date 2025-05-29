import cv2
import face_recognition
import time

time1 = time.time()

image = cv2.imread("images/elon_musk.jpg")
# faceLocs = face_recognition.face_locations(image, model="hog")
faceLocs = face_recognition.face_locations(image, model="cnn")
color = (0,0,255)


for index, faceLoc in enumerate(faceLocs):
    topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
    detectedFaces = image[topLeftY:bottomRightY, topLeftX:bottomRightX]

    cv2.rectangle(image, (topLeftX,topLeftY), (bottomRightX,bottomRightY), color,1)
    #cv2.imwrite("images/croppedFaces.jpg", detectedFaces)
    
    cv2.imshow("Cropped Fce", detectedFaces)
    cv2.imshow("Test Image", image)

time2 = time.time()

# speed = time2 - time1
speedCNN = time2 - time1
print(speedCNN)









































































