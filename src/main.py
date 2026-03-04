from preprocessing.image_preprocessor import ImagePreprocessor
import cv2




prep = ImagePreprocessor()

prep.prep_im_batch()
i = prep.im_tensor[0].numpy()
# i = i.reshape(1, i.shape[0], i.shape[1])
print(i.shape)
cv2.imshow("window",i)
cv2.waitKey(0)