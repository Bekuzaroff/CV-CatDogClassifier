from models.lenet5 import LeNet5
from models.resnet50 import ResNet50
from preprocessing.image_preprocessor import ImagePreprocessor


if __name__ == '__main__':
    lenet_model = LeNet5(input_shape=(32, 32, 1), batch_size=32) # first cv model (lenet architecture)
    resnet50 = ResNet50(input_shape=(224, 224, 3), batch_size=32)

    prep = ImagePreprocessor(batch_size=32) # custom preproc class

    lenet_model.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    resnet50.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    k = 0
    # making batchs and training model on them ->
    for batch in prep.batch_generator("/data/train/", 224):
        train_x = batch[0] # images in array in batch array[3D - (w, h, channel)]
        train_y = batch[1] # labels in batch (vector)

        resnet50.train_on_batch(train_x, train_y)
        k += 1
        
        print(k)
        

    
    for batch in prep.batch_generator("/data/val/", 224):
        test_x = batch[0] # images in array in batch array[3D - (w, h, channel)]

        predicts = resnet50.predict_on_batch(test_x)
        pred_classes = (predicts > 0.5).astype(int)
        
        print(pred_classes, batch[1])
        break

