from sklearn.metrics import accuracy_score, precision_score, recall_score

from models.resnet50 import ResNet50
from preprocessing.image_preprocessor import ImagePreprocessor


if __name__ == '__main__':
    resnet50 = ResNet50(input_shape=(224, 224, 3), batch_size=32)

    prep = ImagePreprocessor(batch_size=32) # custom preproc class
    resnet50.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    k = 0
    # making batchs and training model on them ->
    for batch in prep.batch_generator("/data/train/", 224):
        train_x = batch[0] # images in array in batch array[3D - (w, h, channel)]
        train_y = batch[1] # labels in batch (vector)

        resnet50.train_on_batch(train_x, train_y)
        k += 1
        
        print(k)
        
    k = 0
    recalls = []
    precisions = []
    for batch in prep.batch_generator("/data/val/", 224):
        test_x = batch[0] # images in array in batch array[3D - (w, h, channel)]

        predicts = resnet50.predict_on_batch(test_x)
        pred_classes = (predicts > 0.5).astype(int)
        y_true = batch[1]

        recalls.append(recall_score(y_true, pred_classes))
        precisions.append(precision_score(y_true, pred_classes))
        k+=1
        print(k)

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    print(avg_recall)
    print(avg_precision)

    if avg_precision > 0.96 and avg_precision > 0.96:
        resnet50.model.save("my_model.keras")
