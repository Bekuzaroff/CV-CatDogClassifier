import tensorflow as tf
import keras
class Model1():
    def __init__(self, input_shape=(28, 28, 1), batch_size=32):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = keras.Sequential([
            keras.layers.Input(shape=input_shape, batch_size=batch_size),
            keras.layers.Conv2D(28, kernel_size=5, activation="tanh", padding="same"),
            keras.layers.AvgPool2D(2),
            keras.layers.Conv2D(10, kernel_size=5, activation="tanh", padding="same"),
            keras.layers.AvgPool2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(84, activation="tanh"),
            keras.layers.Dense(1, activation="sigmoid")
        ])
    
    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def predict_on_batch(self, x):
        return self.model.predict_on_batch(x)


