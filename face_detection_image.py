import cv2
import matplotlib.pyplot as plt

test_img_1 = plt.imread("images/elon_musk.jpg")
test_img_2 = plt.imread("images/trump.jpg")


face_cascade = cv2.CascadeClassifier("datasets/haarcascade_frontalface_default.xml")

test_img_1 = cv2.cvtColor(test_img_1, cv2.COLOR_BGR2RGB)
test_img_2 = cv2.cvtColor(test_img_2, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(12,8))  #plt resimleri rgb olarak acar
# plt.imshow(test_img_1)
# plt.show()


gray_test_img_1 = cv2.cvtColor(test_img_1, cv2.COLOR_BGR2GRAY)
gray_test_img_2 = cv2.cvtColor(test_img_2, cv2.COLOR_BGR2GRAY)



faces_test_img_1 = face_cascade.detectMultiScale(gray_test_img_1, 1.3, 3) #x,y,w,h
faces_test_img_2 = face_cascade.detectMultiScale(gray_test_img_2, 1.1, 2)

for (x,y,w,h) in faces_test_img_1:
    cv2.rectangle(test_img_1, (x,y), (x+w, y+h), (255,148,50),3)
    

for (x1,y1,w1,h1) in faces_test_img_2:
    cv2.rectangle(test_img_2, (x1,y1), (x1+w1, y1+h1), (255,148,50),3)



cv2.imshow("Elon Musk", test_img_1)   #cv2 resimleri bgr olarak acar
cv2.imshow("Donald Trump", test_img_2)  






























































