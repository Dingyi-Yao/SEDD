import torch
import torch.nn as nn
import time
import scipy.io
import torch.amp as amp
import os
import numpy as np
from metric import compute_mmd
def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def gram_matrix_batch(tensor):
    # tensor: shape [batch_size, d, h, w]
    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(batch_size, d, h * w)
    gram = torch.bmm(tensor, tensor.transpose(1, 2))
    return gram

def extract_triangle(gram):
    indices = torch.triu_indices(gram.size(1), gram.size(2))
    return gram[:, indices[0], indices[1]]

def ModelTrain(max_num_epoch, FeatureExtractor, StyleExtractor, train_loader_list, eval_loader_list, optimizer_feat, optimizer_style, batch_size, device, lossfunc_style,scheduler_feat, scheduler_style, miner, scale_size, cal_loss_margin, lossfunc_center, center_lambda, optimizer_center, scheduler_center):

    """
    num_epoch: number of epochs
    model: model to train
    train_data: input data
    target: target data
    return: 
    model: trained model
    """
    print_margin = 100

    for epoch in range(max_num_epoch):
        print_loss = 0
        loss_calculation_count = 0
        train_images_city_loader_iter = enumerate(train_loader_list[0])
        train_images_kitti_loader_iter = enumerate(train_loader_list[1])
        train_images_vkitti1_1_loader_iter = enumerate(train_loader_list[2])
        train_images_vkitti1_2_loader_iter = enumerate(train_loader_list[3])
        train_images_vkitti1_3_loader_iter = enumerate(train_loader_list[4])
        train_images_vkitti2_1_loader_iter = enumerate(train_loader_list[5])
        train_images_vkitti2_2_loader_iter = enumerate(train_loader_list[6])
        train_images_vkitti2_3_loader_iter = enumerate(train_loader_list[7])

        iter_list = [train_images_city_loader_iter,
                    train_images_kitti_loader_iter, 
                    train_images_vkitti1_1_loader_iter, 
                    train_images_vkitti1_2_loader_iter, 
                    train_images_vkitti1_3_loader_iter, 
                    train_images_vkitti2_1_loader_iter, 
                    train_images_vkitti2_2_loader_iter, 
                    train_images_vkitti2_3_loader_iter]
        print_loss = 0
        print_style_loss = 0
        print_center_loss = 0

        style_embeddings_list = []
        style_emb_labels_list = []
        step_per_epoch = len(train_loader_list[0])
        for step in range(step_per_epoch):
            
            FeatureExtractor.train()
            StyleExtractor.train()


            conditions = [
                (0, 0),
                (1, 0),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 2),
                (6, 2),
                (7, 2)
            ]
            
            random_scale_idx_flag = 0
            for idx, label in conditions:
                _, img = next(iter_list[idx])
                img = img.to(device)
                feat_img = FeatureExtractor(img)
                feat_img_scaled = feat_img

                gram = gram_matrix_batch(feat_img_scaled)

                vector_gram = extract_triangle(gram)


                vector_gram = (vector_gram - vector_gram.mean()) / (vector_gram.std() + 1e-10)
                style_emb = StyleExtractor(vector_gram)
                
                style_embeddings_list.append(style_emb)
                style_emb_labels_list.append(torch.tensor(label, dtype=torch.long).to(device))
            

            if (step % cal_loss_margin == 0 or step == step_per_epoch - 1) and step != 0:
                optimizer_feat.zero_grad()
                optimizer_style.zero_grad()
                optimizer_center.zero_grad()

                style_embeddings = torch.cat(style_embeddings_list, dim=0)

                style_emb_labels =  torch.stack(style_emb_labels_list)
                

                center_loss = lossfunc_center(style_embeddings, style_emb_labels)
       
                hard_pairs = miner(style_embeddings, style_emb_labels)
                style_loss = lossfunc_style(style_embeddings, style_emb_labels, hard_pairs)
                
                total_loss = style_loss  + center_loss* center_lambda

                total_loss.backward()
                optimizer_style.step()
                optimizer_feat.step()
                optimizer_center.step()

                print_loss += total_loss.item()
                print_style_loss += style_loss.item()
                print_center_loss += center_loss.item()
                loss_calculation_count += 1

                style_embeddings_list = []
                style_emb_labels_list = []

                scheduler_center.step()

            if step % print_margin == 0 and step != 0:
                average_total_loss = print_loss / loss_calculation_count if loss_calculation_count > 0 else 0
                average_style_loss = print_style_loss / loss_calculation_count if loss_calculation_count > 0 else 0
                average_center_loss = print_center_loss / loss_calculation_count if loss_calculation_count > 0 else 0
                print('time: {}, Epoch [{}], Step [{}/{}], Average Total Loss: {:.8f}, Average Style Loss: {:.8f}, Average Center Loss: {:.8f}'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch+1, step, step_per_epoch, average_total_loss, average_style_loss, average_center_loss))
                print_loss = 0
                print_style_loss = 0
                print_center_loss = 0
                loss_calculation_count = 0 

        ModelEval(epoch, FeatureExtractor, StyleExtractor, eval_loader_list, batch_size, device, lossfunc_style, scale_size, miner, cal_loss_margin)
        print('feat_lr:', optimizer_feat.param_groups[0]['lr'])
        print('style_lr:', optimizer_style.param_groups[0]['lr'])
        print('center_lr:', optimizer_center.param_groups[0]['lr'])
    return FeatureExtractor, StyleExtractor


def ModelEval(epoch, FeatureExtractor, StyleExtractor, eval_loader_list, batch_size, device, lossfunc_style, scale_size, miner, cal_loss_margin):
    """
    model: model to evaluate
    test_data: input data
    target: target data
    return:
    loss: loss of the model
    """
    FeatureExtractor.eval()
    StyleExtractor.eval()
    style_emb_list = []
    label_list = []
    with torch.no_grad():
        eval_images_city_loader_iter = enumerate(eval_loader_list[0])
        eval_images_kitti_loader_iter = enumerate(eval_loader_list[1])
        eval_images_vkitti1_1_loader_iter = enumerate(eval_loader_list[2])
        eval_images_vkitti1_2_loader_iter = enumerate(eval_loader_list[3])
        eval_images_vkitti1_3_loader_iter = enumerate(eval_loader_list[4])
        eval_images_vkitti2_1_loader_iter = enumerate(eval_loader_list[5])
        eval_images_vkitti2_2_loader_iter = enumerate(eval_loader_list[6])
        eval_images_vkitti2_3_loader_iter = enumerate(eval_loader_list[7])

        iter_list = [eval_images_city_loader_iter,
                    eval_images_kitti_loader_iter,
                    eval_images_vkitti1_1_loader_iter,
                    eval_images_vkitti1_2_loader_iter,
                    eval_images_vkitti1_3_loader_iter,
                    eval_images_vkitti2_1_loader_iter,
                    eval_images_vkitti2_2_loader_iter,
                    eval_images_vkitti2_3_loader_iter]
        

        step_per_epoch = len(eval_loader_list[0])
        for step in range(step_per_epoch):
            

            conditions = [
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7)
            ]
            
            random_scale_idx_flag = 0
            for idx, label in conditions:
                _, img = next(iter_list[idx])
                img = img.to(device)
                feat_img = FeatureExtractor(img)
                feat_img_scaled = feat_img

                gram = gram_matrix_batch(feat_img_scaled)

                vector_gram = extract_triangle(gram)
                
                vector_gram = (vector_gram - vector_gram.mean()) / (vector_gram.std() + 1e-10)

                style_emb = StyleExtractor(vector_gram)
                
                style_emb_list.append(style_emb.detach().cpu().numpy())
                label_list.append(label)
                torch.cuda.empty_cache()

    style_emb_np = np.array(style_emb_list).squeeze()
    label_np = np.array(label_list)
    class_1 = style_emb_np[np.isin(label_np, [0, 1])]
    class_2 = style_emb_np[np.isin(label_np, [2, 3, 4])]
    class_3 = style_emb_np[np.isin(label_np, [5, 6, 7])]
    class_1_0 = style_emb_np[np.isin(label_np, [0])]
    class_1_1 = style_emb_np[np.isin(label_np, [1])]

    center_1 = np.mean(class_1, axis=0)
    center_2 = np.mean(class_2, axis=0)
    center_3 = np.mean(class_3, axis=0)
    center_1_0 = np.mean(class_1_0, axis=0)
    center_1_1 = np.mean(class_1_1, axis=0)

    distance_center_2_1 = np.linalg.norm(center_2 - center_1)
    print("discenter - vkitti1 vs real:",distance_center_2_1)

    distance_center_3_1 = np.linalg.norm(center_3 - center_1)
    print("discenter - vkitti2 vs real:",distance_center_3_1)

    distance_center_inter = np.linalg.norm(center_1_0 - center_1_1)
    print("discenter - city vs kitti:",distance_center_inter)
    var_1 = np.mean(np.linalg.norm(class_1 - center_1, axis=1))
    var_2 = np.mean(np.linalg.norm(class_2 - center_2, axis=1))
    var_3 = np.mean(np.linalg.norm(class_3 - center_3, axis=1))
    print("var - real:", var_1)
    print("var - vkitti1:", var_2)
    print("var - vkitti2:", var_3)

    cos_1 = np.mean(np.dot(class_1, center_1) / (np.linalg.norm(class_1, axis=1) * np.linalg.norm(center_1)))
    cos_2 = np.mean(np.dot(class_2, center_2) / (np.linalg.norm(class_2, axis=1) * np.linalg.norm(center_2)))
    cos_3 = np.mean(np.dot(class_3, center_3) / (np.linalg.norm(class_3, axis=1) * np.linalg.norm(center_3)))
    print("cos - real:", cos_1)
    print("cos - vkitti1:", cos_2)
    print("cos - vkitti2:", cos_3)

    mmd_2_1 = compute_mmd(class_1, class_2)
    print("mmd - vkitti1 vs real:", mmd_2_1)
    mmd_3_1 = compute_mmd(class_1, class_3)
    print("mmd - vkitti2 vs real:", mmd_3_1)
    mmd_inter = compute_mmd(class_1_0, class_1_1)
    print("mmd - city vs kitti:", mmd_inter)
