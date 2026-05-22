import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torchvision import transforms

from dataloader import get_data_loaders
from dataloader_newdataset import get_data_loaders_newdataset
from metric import compute_mmd
from featextractor import my_resnet18
from styleextractor import StyleExtractor_conv1

plt.rc('font', family='Times New Roman')

np.random.seed(1234)
torch.manual_seed(1234)
random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.backends.cudnn.deterministic = True


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
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


# Paths can be overridden without editing this script, for example:
# SEDD_DATA_ROOT=/path/to/data SEDD_DEVICE=cuda:0 python visstylecode.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
model_path = os.environ.get(
    'SEDD_MODEL_DIR',
    os.path.join(PROJECT_ROOT, 'result_model', 'final'),
)
output_dir = os.environ.get(
    'SEDD_OUTPUT_DIR',
    os.path.join(PROJECT_ROOT, 'result_stylecode_newtrans'),
)
save_path = os.path.join(output_dir, 'final_stylecode.pdf')
base_path = os.environ.get('SEDD_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data'))
eval_cityset_path = os.environ.get(
    'SEDD_CITYSCAPES_VAL_DIR',
    os.path.join(PROJECT_ROOT, 'project/domalign/data/cityscapes/leftImg8bit/val'),
)
os.makedirs(output_dir, exist_ok=True)

t_start = time.time()
device = torch.device(os.environ.get('SEDD_DEVICE', 'cuda'))

# Load the trained feature extractor and style extractor.
FeatureExtractor = my_resnet18().to(device)
StyleExtractor = StyleExtractor_conv1(2080).to(device)

FeatureExtractor.load_state_dict(
    torch.load(
        os.path.join(model_path, 'FeatureExtractor.pth'),
        map_location=device,
        weights_only=True,
    )
)
StyleExtractor.load_state_dict(
    torch.load(
        os.path.join(model_path, 'StyleExtractor.pth'),
        map_location=device,
        weights_only=True,
    )
)

postfix_list = ['clone', 'morning', 'overcast']
seq_num = ['0001', '0002', '0006', '0018', '0020']
batch_size = 1
eval_cityset_postfix_list = ['frankfurt', 'lindau', 'munster']
eval_random_select_num = 426

transform_eval = transforms.Compose([
    transforms.RandomCrop((370, 900)),
    transforms.CenterCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_city_loader = get_data_loaders_newdataset(
    eval_cityset_path,
    eval_cityset_postfix_list,
    transform_eval,
    batch_size,
    eval_random_select_num,
)

eval_images_kitti_loader, eval_images_vkitti1_1_loader, \
eval_images_vkitti1_2_loader, eval_images_vkitti1_3_loader, \
eval_images_vkitti2_1_loader, eval_images_vkitti2_2_loader, \
eval_images_vkitti2_3_loader = get_data_loaders(
    postfix_list,
    base_path,
    seq_num,
    mode='eval',
    batch_size=batch_size,
    transform=transform_eval,
    train_ratio=0.6,
    val_ratio=0.2,
)

eval_loader_list = [
    eval_city_loader,
    eval_images_kitti_loader,
    eval_images_vkitti1_1_loader,
    eval_images_vkitti1_2_loader,
    eval_images_vkitti1_3_loader,
    eval_images_vkitti2_1_loader,
    eval_images_vkitti2_2_loader,
    eval_images_vkitti2_3_loader,
]

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

    iter_list = [
        eval_images_city_loader_iter,
        eval_images_kitti_loader_iter,
        eval_images_vkitti1_1_loader_iter,
        eval_images_vkitti1_2_loader_iter,
        eval_images_vkitti1_3_loader_iter,
        eval_images_vkitti2_1_loader_iter,
        eval_images_vkitti2_2_loader_iter,
        eval_images_vkitti2_3_loader_iter,
    ]

    step_per_epoch = len(eval_loader_list[0])
    for step in range(step_per_epoch):
        # Dataset labels: 0 cityscapes, 1 KITTI, 2-4 VKITTI, 5-7 VKITTI2.
        conditions = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
        ]

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

# Group labels into real, VKITTI, and VKITTI2 domains.
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
print("discenter - vkitti1 vs real:", distance_center_2_1)

distance_center_3_1 = np.linalg.norm(center_3 - center_1)
print("discenter - vkitti2 vs real:", distance_center_3_1)

distance_center_inter = np.linalg.norm(center_1_0 - center_1_1)
print("discenter - city vs kitti:", distance_center_inter)

var_1 = np.mean(np.linalg.norm(class_1 - center_1, axis=1))
var_2 = np.mean(np.linalg.norm(class_2 - center_2, axis=1))
var_3 = np.mean(np.linalg.norm(class_3 - center_3, axis=1))
print("var - kitti:", var_1)
print("var - vkitti1:", var_2)
print("var - vkitti2:", var_3)


mmd_2_1 = compute_mmd(class_1, class_2)
print("mmd - vkitti1 vs real:", mmd_2_1)
mmd_3_1 = compute_mmd(class_1, class_3)
print("mmd - vkitti2 vs real:", mmd_3_1)
mmd_inter = compute_mmd(class_1_0, class_1_1)
print("mmd - city vs kitti:", mmd_inter)

tsne = TSNE(n_components=2, init='pca', random_state=0)
style_emb_tsne = tsne.fit_transform(style_emb_np)

# Plot t-SNE visualization of style embeddings.
plt.figure(figsize=(7, 6), dpi=600)
plt.scatter(style_emb_tsne[label_np==0, 0], style_emb_tsne[label_np==0, 1], s=20, c='#2E5A87', marker='x', label='cityscapes')
plt.scatter(style_emb_tsne[label_np==1, 0], style_emb_tsne[label_np==1, 1], s=20, c='#2E5A87', marker='o', label='kitti')
plt.scatter(style_emb_tsne[label_np==2, 0], style_emb_tsne[label_np==2, 1], s=20, c='#E69F37', marker='o', label='vkitti1_cond1')
plt.scatter(style_emb_tsne[label_np==3, 0], style_emb_tsne[label_np==3, 1], s=20, c='#E69F37', marker='x', label='vkitti1_cond2')
plt.scatter(style_emb_tsne[label_np==4, 0], style_emb_tsne[label_np==4, 1], s=20, c='#E69F37', marker='^', label='vkitti1_cond3')
plt.scatter(style_emb_tsne[label_np==5, 0], style_emb_tsne[label_np==5, 1], s=20, c='#C43D3D', marker='o', label='vkitti2_cond1')
plt.scatter(style_emb_tsne[label_np==6, 0], style_emb_tsne[label_np==6, 1], s=20, c='#C43D3D', marker='x', label='vkitti2_cond2')
plt.scatter(style_emb_tsne[label_np==7, 0], style_emb_tsne[label_np==7, 1], s=20, c='#C43D3D', marker='^', label='vkitti2_cond3')
plt.legend(fontsize=16,
           labelspacing=0.3,
           handletextpad=0.1,
           borderaxespad=0.1,
           borderpad=0.1,
           columnspacing=0.1,
           handlelength=1)

print(f'Model path: {model_path}')
print(f'Save path: {save_path}')
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
plt.savefig(save_path)
plt.close()

t_end = time.time()
print(f'Elapsed time: {t_end - t_start} s')
