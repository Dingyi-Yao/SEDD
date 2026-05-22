#coding:utf-8
import torch
import torch.nn as nn
import numpy as np
from savelog import Logger
import sys,time
import os
from dataloader import get_data_loaders
from dataloader_newdataset import get_data_loaders_newdataset
from torchvision import transforms, models
from featextractor import my_resnet18, my_resnet50, my_resnet152
from styleextractor import StyleExtractor_conv1
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from modeltrain import ModelTrain
from loss import CenterLoss
import random


import argparse
parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('--temperature', type=float, required=True, help='temperature')
parser.add_argument('--lr', type=float, required=True, help='Learning rate')
parser.add_argument('--labmda', type=float, required=True, help='center_lambda')
args = parser.parse_args()

sys.stdout = Logger(sys.stdout, output_dir='./log/')

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True


max_num_epoch = 4
batch_size = 1
temperature = args.temperature
lr_style = args.lr
lr_feat = args.lr
lr_center = 10
margin = 0.01
extractor_name = 'resnet18'
scale_size = 256
cal_loss_margin = 100
num_classes = 3 
feat_dim = 64
center_lambda = args.labmda



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

postfix_list = ['clone', 'morning', 'overcast']
seq_num = ['0001', '0002', '0006', '0018', '0020']

base_path = './data'
train_newdataset_path = './data/cityscapes/leftImg8bit/train'
train_newdataset_postfix_list = ['aachen', 'bochum', 
                           'bremen', 'cologne', 
                           'darmstadt', 'dusseldorf', 
                           'erfurt', 'hamburg', 'hanover', 
                           'jena', 'krefeld', 'monchengladbach', 
                           'strasbourg', 'stuttgart', 'tubingen', 
                           'ulm', 'weimar', 'zurich']
eval_newdataset_path = './data/cityscapes/leftImg8bit/val'
eval_newdataset_postfix_list = ['frankfurt', 'lindau', 'munster']
train_random_select_num = 1275
eval_random_select_num = 426

transform_train = transforms.Compose([
                                transforms.RandomCrop((370, 900)),
                                transforms.CenterCrop((224, 224)),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])


transform_eval = transforms.Compose([
                                transforms.RandomCrop((370, 900)),
                                transforms.CenterCrop((224, 224)),
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])

print('max_num_epoch:', max_num_epoch)
print('batchsize:', batch_size)
print('lr_style:', lr_style)
print('lr_feat:', lr_feat)
print('temperature:', temperature)  
print('extractor_name',extractor_name)
print('lambda:',center_lambda)  
print('lr_center:',lr_center)
print("-------------------------loading train data-------------------------")

train_newdata_loader = get_data_loaders_newdataset(train_newdataset_path, 
                                               train_newdataset_postfix_list, 
                                               transform_train,
                                               batch_size, 
                                               train_random_select_num)


train_images_kitti_loader, train_images_vkitti1_1_loader, \
train_images_vkitti1_2_loader, train_images_vkitti1_3_loader, \
train_images_vkitti2_1_loader, train_images_vkitti2_2_loader, \
train_images_vkitti2_3_loader= get_data_loaders(postfix_list, 
                                            base_path, 
                                            seq_num, 
                                            mode = 'train',
                                            batch_size = batch_size, 
                                            transform=transform_train,
                                            train_ratio=0.6, 
                                            val_ratio=0.2) 

print("-------------------------loading eval data-------------------------")
eval_newdata_loader = get_data_loaders_newdataset(eval_newdataset_path,
                                                  eval_newdataset_postfix_list,
                                                    transform_eval,
                                                    batch_size,
                                                    eval_random_select_num)

eval_images_kitti_loader, eval_images_vkitti1_1_loader, \
eval_images_vkitti1_2_loader, eval_images_vkitti1_3_loader, \
eval_images_vkitti2_1_loader, eval_images_vkitti2_2_loader, \
eval_images_vkitti2_3_loader= get_data_loaders(postfix_list,
                                            base_path, 
                                            seq_num, 
                                            mode = 'eval',
                                            batch_size = batch_size, 
                                            transform=transform_eval,
                                            train_ratio=0.6, 
                                            val_ratio=0.2)
print("-------------------------load data done-------------------------")

FeatureExtractor = my_resnet18(weights="IMAGENET1K_V1")

StyleExtractor = StyleExtractor_conv1(2080)

FeatureExtractor = FeatureExtractor.to(device)
StyleExtractor = StyleExtractor.to(device)



optimizer_style = torch.optim.SGD([p for p in StyleExtractor.parameters() if p.requires_grad == True], lr=lr_style,weight_decay=1e-4, momentum=0.9)
optimizer_feat = torch.optim.SGD([p for p in FeatureExtractor.parameters() if p.requires_grad == True], lr=lr_feat,weight_decay=1e-4, momentum=0.9)

scheduler_feat = torch.optim.lr_scheduler.MultiStepLR(optimizer_feat, milestones=[150, 200], gamma=0.1)
scheduler_style = torch.optim.lr_scheduler.MultiStepLR(optimizer_style, milestones=[150, 200], gamma=0.1)

miner = miners.BatchEasyHardMiner(
        pos_strategy="hard",
        neg_strategy="hard",
    )

lossfunc_style =  losses.NTXentLoss(temperature)

lossfunc_center = CenterLoss(num_classes, feat_dim, device)



optimizer_center = torch.optim.SGD(lossfunc_center.parameters(), lr= lr_center,weight_decay=1e-4, momentum=0.9)

lambda_func = lambda step: 0.9 ** step
scheduler_center = torch.optim.lr_scheduler.LambdaLR(optimizer_center, lr_lambda=lambda_func)

train_loader_list = [train_newdata_loader,
            train_images_kitti_loader, 
            train_images_vkitti1_1_loader, 
            train_images_vkitti1_2_loader, 
            train_images_vkitti1_3_loader, 
            train_images_vkitti2_1_loader, 
            train_images_vkitti2_2_loader, 
            train_images_vkitti2_3_loader]
eval_loader_list = [eval_newdata_loader,
                eval_images_kitti_loader,
                eval_images_vkitti1_1_loader,
                eval_images_vkitti1_2_loader,
                eval_images_vkitti1_3_loader,
                eval_images_vkitti2_1_loader,
                eval_images_vkitti2_2_loader,
                eval_images_vkitti2_3_loader]


FeatureExtractor, StyleExtractor = ModelTrain(max_num_epoch, FeatureExtractor, StyleExtractor, 
                                            train_loader_list, eval_loader_list, optimizer_feat, optimizer_style, 
                                            batch_size, device, lossfunc_style,scheduler_feat, 
                                            scheduler_style, miner, scale_size, cal_loss_margin,
                                            lossfunc_center, center_lambda, optimizer_center, scheduler_center)

save_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
print('save_time',save_time)
save_file_folder = './result_model/'+ save_time
if not os.path.exists(save_file_folder):
    os.makedirs(save_file_folder)

torch.save(FeatureExtractor.state_dict(), save_file_folder + '/FeatureExtractor.pth')
torch.save(StyleExtractor.state_dict(), save_file_folder + '/StyleExtractor.pth')
