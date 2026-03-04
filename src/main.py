from preprocessing.image_preprocessor import ImagePreprocessor
import cv2




prep = ImagePreprocessor()

b = []
for batch in prep.batch_generator("/data/train/"):
    b = batch
    break
print(b[1][0])
cv2.imshow("window", b[0][0])
cv2.waitKey(0)