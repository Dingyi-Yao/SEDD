import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import time
import numpy as np


np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True

class NewDataset(Dataset):
    def __init__(self, root_dir, transform, postfix_list, random_select_num):
        self.root_dir = root_dir
        self.transform = transform
        self.postfix_list = postfix_list
        self.random_select_num = random_select_num
        self.image_paths = self._get_image_paths()


    def _get_image_paths(self):
        image_paths = []
        for folder in self.postfix_list:
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                all_images = os.listdir(folder_path)
                for image_name in all_images:
                    image_path = os.path.join(folder_path, image_name)
                    image_paths.append(image_path)
        if self.random_select_num != -1:
            image_paths = random.sample(image_paths, self.random_select_num)
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image

def get_data_loaders_newdataset(path, postfix_list, transform, batch_size, random_select_num):
    new_dataset = NewDataset(path, transform, postfix_list, random_select_num)
    new_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return new_loader

