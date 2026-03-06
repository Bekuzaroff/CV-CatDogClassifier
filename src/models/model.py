'''
Basic class of models
''' # mark have to add batch size here later and change all from keras to tf
class Model(): 
    def __init__(self, input_shape, batch_size):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.model = None
    
    def train_on_batch(self, x, y):
        pass
    
    def predict_on_batch(self, x):
        pass
