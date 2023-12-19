import torch
import torch.nn as nn
from torch.utils.data import random_split
import torchvision.transforms.functional as TF
from torchvision.models import mobilenet_v2


class CustomCrop:
    def __init__(self, top, left, height, width):
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

class MPFCA(nn.Module):
    def __init__(self, feature_dim):
        super(MPFCA, self).__init__()
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
        attn1 = self.attention_net1(x1)
        attn2 = self.attention_net2(x2)

        weighted_x1 = attn1 * x1
        weighted_x2 = attn2 * x2

        combined_feature = weighted_x1 + weighted_x2
        return combined_feature

class FusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FusionModel, self).__init__()
        self.model1 = MobileNetModel()
        self.model2 = MobileNetModel()
        self.attention = MPFCA(feature_dim=32)
        self.fc = nn.Linear(32, num_classes)

    def extra_repr(self):
        model1_repr = self.model1.extra_repr()
        model2_repr = self.model2.extra_repr()
        attention_repr = "MPFCA"
        classifier_repr = f"Final classifier: Linear(in_features=32, out_features={self.fc.out_features})"
        return f"{model1_repr}\n{model2_repr}\n{attention_repr}\n{classifier_repr}"

    def forward(self, x1, x2):
        out1 = self.model1(x1)
        out2 = self.model2(x2)

        combined_feature = self.attention(out1, out2)
        x = self.fc(combined_feature)
        return x

class MobileNetModel(nn.Module):
    def __init__(self, features_num=32):
        super(MobileNetModel, self).__init__()
        self.mobilenet = mobilenet_v2(pretrained=True)
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_features, features_num)

    def forward(self, x):
        features = self.mobilenet(x)
        return features


