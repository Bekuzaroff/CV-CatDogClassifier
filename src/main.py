import keras


from models.lenet5 import LeNet5
from preprocessing.image_preprocessor import ImagePreprocessor
import cv2


m1 = LeNet5()
prep = ImagePreprocessor()

b = []

m1.model.compile(optimizer="adam", loss="binary_crossentropy")

for batch in prep.batch_generator("/data/train/", 32):
    trainx = batch[0]
    trainy = batch[1]
    b = trainx
    m1.train_on_batch(trainx, trainy)
    break

preds = m1.predict_on_batch(b)
print(preds)
