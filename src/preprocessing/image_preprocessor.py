import cv2 as cv
import numpy as np

class ImagePreprocessor():
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def read_image(self, im_abs_path: str, channels: bool = False):
        '''
        just reading image and checking if it exists:
            params:
                im_abs_path: str - abs path to image
            output type: Matlike | None
        '''
        try:
            im = cv.imread(im_abs_path, int(channels))

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
         # Resize
        preped_im = cv.resize(image, (im_size, im_size))
        
        # BGR -> RGB
        preped_im = cv.cvtColor(preped_im, cv.COLOR_BGR2RGB)
        
        # float32 and 0-1
        preped_im = preped_im.astype(np.float32) / 255.0
        
        # Normalize
        preped_im = (preped_im - self.mean) / self.std

        return preped_im







       
        


    