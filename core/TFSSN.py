import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision.transforms.functional as TF
import pandas as pd
import os
import json
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
from scipy.signal import stft, kaiser
import numpy as np

class SWJTU_Gear_Data:
    def __init__(self, filename, fs = 12800, file_path=''):
        self.file_path = file_path
        path = self.file_path +os.sep + filename
        self.df = pd.read_csv(path)
        self.fs = fs

    def plt_stft(self, data_type, save_path, file_name):
        nperseg = 2048  # window size
        noverlap = int(nperseg * 0.75)  # overlap size
        nfft = nperseg  # FFT size
        name_dic = {'in_x':'输入轴x','in_y':'输入轴y','out_x':'输出轴x','out_y':'输出轴y','out_z':'输出轴z'}
        frequencies_stft, times_stft, Zxx = stft(self.df[name_dic[data_type]], fs=self.fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,window=kaiser(nperseg, 15))
        mask = frequencies_stft <= 1500

        # Plotting the spectrogram
        # plt.figure(figsize=(0.65, 4.81))
        plt.figure(figsize=(1.29, 2.09))
        plt.pcolormesh(times_stft, frequencies_stft[mask], 10 * np.log10(np.abs(Zxx[mask, :])), shading='gouraud',
                       cmap='jet')
        plt.ylim([frequencies_stft[mask][0], frequencies_stft[mask][-1]])
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        # plt.show(bbox_inches='tight')
        plt.savefig(save_path+'//'+'{}.png'.format(file_name), dpi=100)
        plt.close()
        plt.clf()

class CustomCrop:
    def __init__(self, top, left, height, width):
        print(f"top: {top}, left: {left}, height: {height}, width: {width}")
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return TF.crop(img, self.top, self.left, self.height, self.width)


class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, positive_label=0, interval=24):
        self.dataset = dataset
        self.positive_label = positive_label
        self.interval = interval

        self.indices = []
        for i, (_, label) in enumerate(dataset):
            if label == self.positive_label and i % self.interval == 0:
                self.indices.append(i)
            elif label != self.positive_label:
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class ImprovedFCAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ImprovedFCAttention, self).__init__()
        self.attention_net1 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        self.attention_net2 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 计算注意力分数
        attn1 = self.attention_net1(x1)
        attn2 = self.attention_net2(x2)

        # 加权特征
        weighted_x1 = attn1 * x1
        weighted_x2 = attn2 * x2

        # 结合特征
        combined_feature = weighted_x1 + weighted_x2
        return combined_feature

class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FusionModel, self).__init__()
        self.model1 = MobileNetModel()
        self.model2 = MobileNetModel()
        self.attention = ImprovedFCAttention(feature_dim=32)  # 假设特征维度为32
        self.fc = nn.Linear(32, num_classes)

    def extra_repr(self):
        model1_repr = self.model1.extra_repr()
        model2_repr = self.model2.extra_repr()
        attention_repr = "Improved FC Attention Mechanism"
        classifier_repr = f"Final classifier: Linear(in_features=32, out_features={self.fc.out_features})"
        return f"{model1_repr}\n{model2_repr}\n{attention_repr}\n{classifier_repr}"

    def forward(self, x1, x2):
        out1 = self.model1(x1)
        out2 = self.model2(x2)

        # 应用改进的注意力机制
        combined_feature = self.attention(out1, out2)

        # 最终分类
        x = self.fc(combined_feature)
        return x

class MobileNetModel(nn.Module):
    def __init__(self, features_num=32):
        super(MobileNetModel, self).__init__()
        # 加载预训练的MobileNetV2
        self.mobilenet = mobilenet_v2(pretrained=True)
        # 替换分类器
        # 提取MobileNetV2最后一个卷积层的输出特征数量
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, features_num)

    def forward(self, x):
        features = self.mobilenet(x)
        # print(features.shape)
        return features


if __name__ == '__main__':
    with open('../conf/2layer_config.json', 'r') as config_file:
        config = json.load(config_file)
    train_test = config["train_test"]
    data_paths = config["data_paths"]
    path_1 = os.path.join(os.path.join(data_paths,train_test),'train')
    path_2 = os.path.join(os.path.join(data_paths,train_test),'test')
    save_path = config["save_path"]
    save_path = os.path.join(save_path, train_test)
    model_save_path = os.path.join(save_path, 'model_dicts')
    csv_save_paths = os.path.join(save_path, 'epoch_data')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(csv_save_paths):
        os.makedirs(csv_save_paths)
    file_name = config["file_name"]
    # 还差裁剪对应频率的图片，需要再添加一个循环，保存的时候，也要把频率包含进去
    frequency_dic = {999:[0,209],1160:[7,32], 440:[128,20], 200:[170,13]}
    path_train = os.path.join(path_1, file_name)
    path_test = os.path.join(path_2, file_name)
    # 数据预处理：您可以根据需要进行调整
    transform_1160  = transforms.Compose([
        CustomCrop(top=frequency_dic[1160][0], left=0, height=frequency_dic[1160][1], width=129),  # 裁剪图片 (这部分被注释掉了，如果您的数据集需要裁剪请打开)
        transforms.Resize((224, 224)),  # 例如，将所有图片都调整到32x32大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_440   = transforms.Compose([
        CustomCrop(top=frequency_dic[440][0], left=0, height=frequency_dic[440][1], width=129),  # 裁剪图片 (这部分被注释掉了，如果您的数据集需要裁剪请打开)
        transforms.Resize((224, 224)),  # 例如，将所有图片都调整到32x32大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

