import cv2 as cv
import os
import numpy as np
import tensorflow as tf
class ImagePreprocessor():
    def __init__(self):
        self.im_tensor = tf.constant([])
        self.batch_size = 32

    def read_image(self, im_name: str):
        try:
            im = cv.imread(im_name, 0)

            if im is None:
                raise Exception("no such image")
            
            return im
            
        except Exception as e:
            msg = e.args[0]
            print(msg)
    
    def im_preprocess(self, image, im_size):
        preped_im = cv.resize(image, (im_size, im_size))
        preped_im = preped_im / 255.0

        return preped_im
    
    def prep_im_batch(self):
        launch_dir = os.getcwd()
        launch_dir = launch_dir.replace("\\", "/")
        launch_dir = launch_dir + "/data/train/"

        f_names = os.listdir(launch_dir)
        buff = []

        for f_name in f_names:
            mat_im = self.read_image(launch_dir + f_name)
            mat_im = self.im_preprocess(mat_im)

            buff.append(mat_im)
        
        self.im_tensor = tf.convert_to_tensor(buff)
    
    def batch_generator(self, data_dir, im_size):
        cur_dir = os.getcwd()
        cur_dir = cur_dir.replace("\\", "/")
        f_names = os.listdir(cur_dir + data_dir)
        batch = []
        batch_y = []
        for f_name in f_names:
            img_path = os.path.join(cur_dir + data_dir, f_name)

            mat_im = self.read_image(img_path)
            mat_im = self.im_preprocess(mat_im, im_size)
            mat_im = mat_im.reshape(im_size, im_size, 1)

            batch.append(mat_im)
            batch_y.append(1 if f_name.split('.')[0] == "dog" else 0)

            if len(batch) == self.batch_size:
                yield (np.array(batch), np.array(batch_y))
                batch = []
        
        if batch:
            yield (np.array(batch), np.array(batch_y))







       
        


    