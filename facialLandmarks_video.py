import cv2
import dlib

cap = cv2.VideoCapture("videos/elon_musk.mp4")


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    
    color = (0,255,0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detector = dlib.get_frontal_face_detector()
    predictor68 = dlib.shape_predictor("datasets/shape_predictor_68_face_landmarks.dat")
    predictor81 = dlib.shape_predictor("datasets/shape_predictor_81_face_landmarks.dat")
    
    faceLocs = detector(frame)
    
    for index, faceLoc in enumerate(faceLocs):
        landmarks = predictor68(gray, faceLoc)
        
        for i in range(0,68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.putText(frame, str(i),(x,y),font, .3, color, 1, cv2.LINE_AA)
            cv2.circle(frame, (x,y), 2, color,-1)

    cv2.imshow("Facial Landmark Points", frame)
    
    if cv2.waitKey(30) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()































































