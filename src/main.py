import keras

from models.model1 import Model1
from preprocessing.image_preprocessor import ImagePreprocessor
import cv2


m1 = Model1()
prep = ImagePreprocessor()

b = []
m1.model.compile(optimizer="sgd", loss="binary_crossentropy")
for batch in prep.batch_generator("/data/train/"):
    trainx = batch[0]
    trainy = batch[1]
    b = trainx
    m1.train_on_batch(trainx, trainy)
    break

preds = m1.predict_on_batch(b)
print(preds)
