import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance # ökid bagintisi kullanmak icin

cap = cv2.VideoCapture(0)

def eyeAspectRatio(eyePoints):
    
    verticalLine1 = distance.euclidean(eyePoints[1], eyePoints[5])
    verticalLine2 = distance.euclidean(eyePoints[2], eyePoints[4])
    
    horizontalLine = distance.euclidean(eyePoints[0], eyePoints[3])
    
    ear = (verticalLine1+verticalLine2)/(2*horizontalLine)
    
    return ear

print("BİLGİLENDİRME... Dosyalar yükleniyor...")
path = "datasets/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

print("BİLGİLENDİRME... Koordinatlar alınıyor...")
(leftStart, leftEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rightStart, rightEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
thresholValue = 0.27
counter = 0
realCounter = 0


print("BİLGİLENDİRME... Kameraya bağlanılıyor...")
print("-----------------------------------------")

while True:
    
    ret,frame = cap.read()
    if ret == False:
        break
    
    frame = cv2.flip(frame,1) # 1 yazmak y axisine göre döndürmek  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    
    for face in faces:
        landmarkPoints = predictor(gray, face)
        landmarkPoints = face_utils.shape_to_np(landmarkPoints)
        
        leftEye = landmarkPoints[leftStart:leftEnd]
        rightEye = landmarkPoints[rightStart:rightEnd]
        
        leftEyeAspectRatio = eyeAspectRatio(leftEye)
        rightEyeAspectRation = eyeAspectRatio(rightEye)
        
        ear = (leftEyeAspectRatio+rightEyeAspectRation)/2
        
        leftConvexHull = cv2.convexHull(leftEye)
        rightConvexHull = cv2.convexHull(rightEye)
        
        cv2.drawContours(frame, [leftConvexHull], -1, color, 1)
        cv2.drawContours(frame, [rightConvexHull], -1, color, 1)
        
        if ear < thresholValue:
            counter += 1
        else:
            if counter >= 3:
                realCounter += 1
                print("BİLGİLENDİRME...{}. Kırpma tespit edildi...".format(realCounter))
                
            counter = 0
            
        cv2.putText(frame, "Blink: {}".format(realCounter),(25,50),font,0.8,color,2)
        cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear),(300,50),font,0.8,color,2)
        
        
    cv2.imshow(" ", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()





















































































