import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Label Encoder
class LabelEncoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(labels)}
    def get_label(self, idx):
        return list(self.labels.keys())[idx]
    def get_idx(self, label):
        return self.labels.get(label)

# Define the model architecture (must match the trained model)
class CheckpointedResNet152(nn.Module):
    def __init__(self, num_classes=101):
        super(CheckpointedResNet152, self).__init__()
        base_model = models.resnet152(weights='IMAGENET1K_V1')
       
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
       
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
       
        x = torch.utils.checkpoint.checkpoint_sequential(self.layer1, 1, x)
        x = torch.utils.checkpoint.checkpoint_sequential(self.layer2, 1, x)
        x = torch.utils.checkpoint.checkpoint_sequential(self.layer3, 1, x)
        x = torch.utils.checkpoint.checkpoint_sequential(self.layer4, 1, x)
       
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load class names (Food-101 has 101 classes)
def load_classes(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().splitlines()
    return classes