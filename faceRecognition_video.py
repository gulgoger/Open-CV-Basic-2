import cv2
import face_recognition
import imutils #(cv2) kütüphanesini tamamlayıcı bir yardımcı kütüphanedir. 



pathTest = "videos/elon_musk.mp4"
color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX


cap = cv2.VideoCapture(pathTest)

pathTrump = "images/trump.jpg"
trumpImage = face_recognition.load_image_file(pathTrump)
trumpImageEncodings = face_recognition.face_encodings(trumpImage)[0]


pathElon = "images/elon_musk.jpg"
elonImage = face_recognition.load_image_file(pathElon)
elonImageEncodings = face_recognition.face_encodings(elonImage)[0]

encodingsList = [trumpImageEncodings, elonImageEncodings]
namesList = ["Donald Trump", "Elon Musk"]



while True:
    
    ret,frame = cap.read()
    
    if ret == False:
        break
    
    rows, columns, channels = frame.shape
    coefficient = 4
    currentColumn = int(columns/coefficient)
    
    frame = imutils.resize(frame, width=currentColumn)
    
    
    faceLocations = face_recognition.face_locations(frame)
    faceEncodings = face_recognition.face_encodings(frame, faceLocations)
    
    for faceLoc, faceEncoding in zip(faceLocations, faceEncodings):
        topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
        matchedFaces = face_recognition.compare_faces(encodingsList, faceEncoding)
        
        name = "Unknown"
        
        if True in matchedFaces:
            matchedIndex = matchedFaces.index(True)
            name = namesList[matchedIndex]
    
        cv2.rectangle(frame, (topLeftX,topLeftY),(bottomRightX,bottomRightY),color,1)
        cv2.putText(frame, name, (topLeftX,topLeftY), font, 1/(coefficient/1.5), color, 1)
    
        cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
















































