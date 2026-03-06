import keras

from models.model import Model


class ResNet50(Model):
    def __init__(self, input_shape=(224, 224, 3), batch_size=32):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.base_model =  keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
        self.base_model.trainable = False
        # lenet5 architecture ->
        self.model = keras.Sequential([
            self.base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
    
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)