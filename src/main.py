from preprocessing.image_preprocessor import ImagePreprocessor
import cv2




prep = ImagePreprocessor()

im = prep.read_image("C:/Users/User/Documents/ml/cat_dog_classifier/data/train/cat.0.jpg")
cv2.imshow("my first image with openCV", im)

cv2.waitKey(0)