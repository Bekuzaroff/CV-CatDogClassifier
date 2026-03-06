import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import random

class ImagePreprocessor():
    def __init__(self):
        self.batch_size = 32

    def read_image(self, im_abs_path: str):
        '''
        just reading image and checking if it exists:
            params:
                im_abs_path: str - abs path to image
            output type: Matlike | None
        '''
        try:
            im = cv.imread(im_abs_path, 0)

            if im is None: # if no image or other type (not image)
                raise Exception("no such image")
            
            return im
            
        except Exception as e:
            msg = e.args[0] # exception message
            print(msg)
    
    def im_preprocess(self, image, im_size):
        '''
        preprocessing img:
            params:
                image: array[2D] - image in iterable type in 2D
                im_size: int - image size we need
        '''
        preped_im = cv.resize(image, (im_size, im_size)) # resizing image
        preped_im = preped_im / 255.0 # scaling all the pixels
        preped_im = preped_im.reshape(im_size, im_size, 1)

        return preped_im
    
    def batch_generator(self, data_dir, im_size, is_test=False):
        '''
        gives one image batch a time
        params:
            data_dir - str (your path with data-set)
            im_size - int (the size of image to resize to)
        return:
            Tuple[Numpy.array] - tuple[0] - images, tuple[1] - labels
        '''
        cur_dir = os.getcwd() # cur dir
        cur_dir = cur_dir.replace("\\", "/") 
        f_names = os.listdir(cur_dir + data_dir)

        batch = [] # batch for images
        batch_y = [] # batch for labels
        
        random.shuffle(f_names)

        for f_name in f_names:
            # join img name to data dir path, reading and preproc img ->
            img_path = os.path.join(cur_dir + data_dir, f_name)

            mat_im = self.read_image(img_path)
            mat_im = self.im_preprocess(mat_im, im_size)

            # adding prepared img and label to list ->
            batch.append(mat_im)
            
            if not is_test:
                batch_y.append(1 if f_name.split('.')[0] == "dog" else 0)

            # if we got the first batch full ->
            if len(batch) == self.batch_size:
                yield (np.array(batch), np.array(batch_y)) # giving one tuple a time
                batch = []
                batch_y = []

        # for least not full batch ->
        if batch:
            yield (np.array(batch), np.array(batch_y))







       
        


    