
import cv2
import face_recognition

pathTest = "videos/elon_musk.mp4"
cap = cv2.VideoCapture(pathTest)

while True:
    
    ret,frame = cap.read()
    if ret == False:
        break
    
    faceLocs = face_recognition.face_locations(frame, model="hog")
    color = (0,0,255)
    
    for index, faceLoc in enumerate(faceLocs):
        topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
        
        detectedFaces = frame[topLeftY:bottomRightY, topLeftX:bottomRightX]
        
        kernelSize = (27,27)
        blurresFaces = cv2.GaussianBlur(detectedFaces, kernelSize, 30)
        frame[topLeftY:bottomRightY, topLeftX:bottomRightX] = blurresFaces
        
        # cv2.rectangle(frame, (topLeftX,topLeftY), (bottomRightX,bottomRightY),color,1)
        
        cv2.imshow("Test Image",frame)
        
    if cv2.waitKey(15) & 0xFF == ord("q"): break


cap.release()
cv2.destroyAllWindows()




































































