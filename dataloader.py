import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
# import time
import numpy as np
# 随机数种子
random.seed(42)

class MyLoadData():
    def __init__(self, dataset_name, postfix_list, base_path, seq_num, transform):
        self.dataset_name = dataset_name
        self.postfix_list = postfix_list
        self.base_path = base_path
        self.seq_num = seq_num
        self.transform = transform
        self.data_paths = []
        self.labels = []
        self.load_data()

    def load_data(self):
        label = {'kitti': 0, 'vkitti1': 1, 'vkitti2': 2}[self.dataset_name]
        if self.dataset_name == 'kitti':
            for seq in self.seq_num:
                dataset_path = os.path.join(self.base_path, self.dataset_name, seq)
                self._load_images_labels(dataset_path, label)
        

        else:
            for postfix in self.postfix_list:
                for seq in self.seq_num:
                    dataset_path = self._get_dataset_path(seq, postfix)
                    self._load_images_labels(dataset_path, label)


    def _get_dataset_path(self, seq, postfix):
        if self.dataset_name == 'vkitti1':
            return os.path.join(self.base_path, self.dataset_name, seq, postfix)
        elif self.dataset_name == 'vkitti2':
            return os.path.join(self.base_path, self.dataset_name, seq, postfix, 'frames/rgb/Camera_0')
    
    def _load_images_labels(self, dataset_path, label):
        files = os.listdir(dataset_path)
        files.sort()
        for filename in files:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dataset_path, filename)
                self.data_paths.append(img_path)
                self.labels.append(label)

class CustomDataset(Dataset):
    def __init__(self, file_path, transform):
        self.file_path = file_path
        self.transform = transform

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        img_name = self.file_path[idx]
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image


def get_split_indices(data_size, train_ratio, val_ratio):

    indices = list(range(data_size))
    random.shuffle(indices)
    
    train_end = int(train_ratio * data_size)
    val_end = train_end + int(val_ratio * data_size)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    return train_indices, val_indices, test_indices

def split_data_kitti(images_kitti, labels_kitti, train_indices, val_indices, test_indices, mode):
    def get_splits(indices):
        images_kitti_con = []

        for i in indices:
            images_kitti_con.append(images_kitti[i])

        return images_kitti_con

    if mode == 'train':
        train_images_kitti = get_splits(train_indices)
        return train_images_kitti
    elif mode == 'val':
        val_images_kitti = get_splits(val_indices)
        return val_images_kitti
    else:
        test_images_kitti = get_splits(test_indices)
        return test_images_kitti

def split_data_virtual_kitti(images_vkitti, labels_vkitti, data_size, train_indices, val_indices, test_indices, mode):

    def get_splits(indices):
        images_vkitti1_con1 = []
        images_vkitti1_con2 = []
        images_vkitti1_con3 = []

        for i in indices:
            images_vkitti1_con1.append(images_vkitti[i])
            images_vkitti1_con2.append(images_vkitti[i + data_size])
            images_vkitti1_con3.append(images_vkitti[i + 2 * data_size])

        return images_vkitti1_con1, images_vkitti1_con2, images_vkitti1_con3


    if mode == 'train':
        train_images_vkitti_1, train_images_vkitti_2,\
        train_images_vkitti_3 = get_splits(train_indices)
        return train_images_vkitti_1, train_images_vkitti_2,\
        train_images_vkitti_3
    elif mode == 'val':
        val_images_vkitti_1, val_images_vkitti_2, \
        val_images_vkitti_3 = get_splits(val_indices)
        return val_images_vkitti_1, val_images_vkitti_2, \
            val_images_vkitti_3
    else:
        test_images_vkitti_1, test_images_vkitti_2, \
        test_images_vkitti_3 = get_splits(test_indices)
        return test_images_vkitti_1, test_images_vkitti_2, \
            test_images_vkitti_3

def get_data_loaders(postfix_list, base_path, seq_num, mode, batch_size, transform, train_ratio=0.6, val_ratio=0.2) :
    

    dataset_kitti = MyLoadData('kitti', postfix_list, base_path, seq_num, transform=transform)
    data_size = len(dataset_kitti.data_paths)

    train_indices, val_indices, test_indices = get_split_indices(data_size, train_ratio, val_ratio)

    if mode == 'train':
        sampler = torch.utils.data.SubsetRandomSampler(np.random.permutation(len(train_indices)))
    elif mode == 'val':
        sampler = torch.utils.data.SubsetRandomSampler(np.random.permutation(len(val_indices)))
    else:
        sampler = torch.utils.data.SubsetRandomSampler(np.random.permutation(len(test_indices)))

    mode_images_kitti_path = split_data_kitti(dataset_kitti.data_paths, dataset_kitti.labels, train_indices, val_indices, test_indices, mode)
    mode_images_kitti = CustomDataset(mode_images_kitti_path, transform=transform)
    mode_images_kitti_loader = DataLoader(mode_images_kitti, batch_size=batch_size, sampler=sampler)

    dataset_vkitti1 = MyLoadData('vkitti1', postfix_list, base_path, seq_num, transform=transform)
    mode_images_vkitti1_1_path, mode_images_vkitti1_2_path, mode_images_vkitti1_3_path = split_data_virtual_kitti(dataset_vkitti1.data_paths, dataset_vkitti1.labels, data_size, train_indices, val_indices, test_indices, mode)
    mode_images_vkitti1_1 = CustomDataset(mode_images_vkitti1_1_path, transform=transform)
    mode_images_vkitti1_2 = CustomDataset(mode_images_vkitti1_2_path, transform=transform)
    mode_images_vkitti1_3 = CustomDataset(mode_images_vkitti1_3_path, transform=transform)

    mode_images_vkitti1_1_loader = DataLoader(mode_images_vkitti1_1, batch_size=batch_size, sampler= sampler)
    mode_images_vkitti1_2_loader = DataLoader(mode_images_vkitti1_2, batch_size=batch_size, sampler= sampler)
    mode_images_vkitti1_3_loader = DataLoader(mode_images_vkitti1_3, batch_size=batch_size, sampler= sampler)

    dataset_vkitti2 = MyLoadData('vkitti2', postfix_list, base_path, seq_num, transform=transform)
    mode_images_vkitti2_1_path, mode_images_vkitti2_2_path, mode_images_vkitti2_3_path = split_data_virtual_kitti(dataset_vkitti2.data_paths, dataset_vkitti2.labels, data_size, train_indices, val_indices, test_indices, mode)
    mode_images_vkitti2_1 = CustomDataset(mode_images_vkitti2_1_path, transform=transform)
    mode_images_vkitti2_2 = CustomDataset(mode_images_vkitti2_2_path, transform=transform)
    mode_images_vkitti2_3 = CustomDataset(mode_images_vkitti2_3_path, transform=transform)

    mode_images_vkitti2_1_loader = DataLoader(mode_images_vkitti2_1, batch_size=batch_size, sampler= sampler)
    mode_images_vkitti2_2_loader = DataLoader(mode_images_vkitti2_2, batch_size=batch_size, sampler= sampler)
    mode_images_vkitti2_3_loader = DataLoader(mode_images_vkitti2_3, batch_size=batch_size, sampler= sampler)
    return mode_images_kitti_loader, mode_images_vkitti1_1_loader, mode_images_vkitti1_2_loader, \
        mode_images_vkitti1_3_loader, mode_images_vkitti2_1_loader, mode_images_vkitti2_2_loader, mode_images_vkitti2_3_loader
