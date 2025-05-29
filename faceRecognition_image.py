import cv2
import face_recognition


pathTest = "images/test.jpg"
color = (0,255,0)
image = cv2.imread(pathTest)

font = cv2.FONT_HERSHEY_SIMPLEX

pathTrump = "images/trump.jpg"
trumpImage = face_recognition.load_image_file(pathTrump)
trumpImageEncodings = face_recognition.face_encodings(trumpImage)[0]


pathElon = "images/elon_musk.jpg"
elonImage = face_recognition.load_image_file(pathElon)
elonImageEncodings = face_recognition.face_encodings(elonImage)[0]

encodingsList = [trumpImageEncodings, elonImageEncodings]
namesList = ["Donald Trump", "Elon Musk"]


testImage = face_recognition.load_image_file(pathTest)
faceLocations = face_recognition.face_locations(testImage)
faceEncodings = face_recognition.face_encodings(testImage,faceLocations)

for faceLoc, faceEncoding in zip(faceLocations, faceEncodings):
    topLeftY, bottomRightX, bottomRightY, topLeftX = faceLoc
    matchedFaces = face_recognition.compare_faces(encodingsList, faceEncoding)
    
    name = "Unknown"
    
    if True in matchedFaces:
        matchedIndex = matchedFaces.index(True)
        name = namesList[matchedIndex]

    cv2.rectangle(image, (topLeftX,topLeftY),(bottomRightX,bottomRightY),color,1)
    cv2.putText(image, name, (topLeftX,topLeftY), font,1,color,1)

    cv2.imshow("Face Recognition", image)


















































