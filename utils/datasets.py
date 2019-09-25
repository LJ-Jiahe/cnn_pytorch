import os

from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, data_dir, input_dir, target_dir, transform=None):
        self.data_dir = data_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        self.input_contents = os.listdir(os.path.join(data_dir, input_dir))
        self.target_contents = os.listdir(os.path.join(data_dir, target_dir))

        self.input_contents.sort(key=lambda x: int(x.split('.')[0]))
        self.target_contents.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.input_contents)

    def __getitem__(self, idx):
        input_image = Image.open(os.path.join(self.data_dir,
                                              self.input_dir,
                                              self.input_contents[idx]))
        target_image = Image.open(os.path.join(self.data_dir,
                                               self.target_dir,
                                               self.target_contents[idx]))

        item = {'input_image': self.transform(input_image),
                'target_image': self.transform(target_image)}

        return item


class Classification_Dataset(Dataset)
    def __init__(self, data_dir, input_dir, target_dir, transform=None):
        
    
    def __getitem__(self):
