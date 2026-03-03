import cv2 as cv
import os

class ImagePreprocessor():
    def __init__(self):
        pass

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
        preped_im = cv.resize(image, im_size)
        preped_im = preped_im / 255.0

        return preped_im
    
    def prep_im_batch():
        # launch_dir = os.get
        pass
        


    