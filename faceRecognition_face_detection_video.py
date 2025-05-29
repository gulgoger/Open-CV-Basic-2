import cv2
import face_recognition

cap = cv2.VideoCapture("videos/elon_musk.mp4")

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    faceLocs = face_recognition.face_locations(frame, model="hog")
    color = (0,0,255)
    
    for index, faceLoc in enumerate(faceLocs):
        topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
        
        # detectedFaces = frame[topLeftY:bottomRightY, topLeftX:bottomRightX]

        cv2.rectangle(frame, (topLeftX,topLeftY), (bottomRightX,bottomRightY),color,1)
        
        # cv2.imshow("Cropped Face", detectedFaces)
        cv2.imshow("Test Image", frame)

    
    if cv2.waitKey(15) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()























































