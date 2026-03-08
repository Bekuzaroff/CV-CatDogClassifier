import os
import random
import torch as t

from torch.utils.data import Dataset

from preprocessing.image_preprocessor import ImagePreprocessor
class MyDataset(Dataset):
    def __init__(self, data_dir, im_size):
        self.file_list = []  # только пути к файлам
        self.labels = []
        
        im_prep = ImagePreprocessor()  # для использования в __getitem__
        self.im_prep = im_prep
        self.im_size = im_size
        
        cur_dir = os.getcwd().replace("\\", "/") 
        f_names = os.listdir(cur_dir + data_dir)
        random.shuffle(f_names)

        for f_name in f_names[:6000]:  # сколько угодно
            self.file_list.append(os.path.join(cur_dir + data_dir, f_name))
            self.labels.append([1] if 'dog' in f_name.lower() else [0])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # Читаем изображение ТОЛЬКО когда оно нужно
        img = self.im_prep.read_image(self.file_list[index], True)
        img = self.im_prep.im_preprocess(img, self.im_size)
        
        x = t.FloatTensor(img)
        x = x.permute(2, 0, 1)
        y = t.LongTensor(self.labels[index])[0]
        return x, y