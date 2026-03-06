from models.lenet5 import LeNet5
from preprocessing.image_preprocessor import ImagePreprocessor


if __name__ == '__main__':
    lenet_model = LeNet5() # first cv model (lenet architecture)
    prep = ImagePreprocessor() # custom preproc class

    lenet_model.model.compile(optimizer="adam", loss="binary_crossentropy")

    # making batchs and training model on them ->
    for batch in prep.batch_generator("/data/train/", 32):
        train_x = batch[0] # images in array in batch array[3D - (w, h, channel)]
        train_y = batch[1] # labels in batch (vector)

        lenet_model.train_on_batch(train_x, train_y)
        break


