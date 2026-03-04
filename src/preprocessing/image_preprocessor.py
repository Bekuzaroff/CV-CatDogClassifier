import cv2 as cv
import os
import tensorflow as tf
class ImagePreprocessor():
    def __init__(self):
        self.im_tensor = tf.constant([])

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
        preped_im = cv.resize(image, (500, 500))
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
            mat_im = self.im_preprocess(mat_im, 200)

            buff.append(mat_im)
        
        self.im_tensor = tf.convert_to_tensor(buff)







       
        


    