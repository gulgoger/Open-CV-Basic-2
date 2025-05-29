import cv2
from skimage.feature import hog
from skimage import exposure

image = cv2.imread("images/elon_musk.jpg", 0)


### STATIC IMAGE ###

# _, hogImage = hog(image, visualize=True)
# rescaledImage = exposure.rescale_intensity(hogImage, in_range=(0,10))

# cv2.imshow("HOG", hogImage)
# cv2.imshow("rescaled Image HOG", rescaledImage)


### REAL TIME ###

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    _, hogImage = hog(frame, visualize=True, channel_axis=-1)
    rescaledImage = exposure.rescale_intensity(hogImage,in_range=(0,10))
    
    cv2.imshow("HOG", rescaledImage)
    
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
















































