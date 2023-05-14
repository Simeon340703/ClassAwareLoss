"""
@c Written by Simegnew Alaba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


class ClassAwareLoss(nn.Module):
    """This code is a PyTorch implementation of ClassAwareLoss used in the
     "Class-aware fish species recognition using deep learning for an imbalanced
      dataset" paper. https://www.mdpi.com/1424-8220/22/21/8268
    """

    def __init__(self, num_classes):
        super(ClassAwareLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, confidence, target):
        cross_entropy_loss = F.cross_entropy(confidence, target, reduction='sum')
        # Count the number of samples for each class
        samples_per_cls = Counter(target.view(-1).tolist())
        # Total number of samples
        total_samples = sum(samples_per_cls.values())
        # Class coefficient for each class
        eta = 4
        class_coeff = 1 - torch.pow(torch.Tensor(list(samples_per_cls.values())) / total_samples, eta)
        # weights for each class
        class_weights = (1 - torch.Tensor(list(samples_per_cls.values())) / total_samples) / class_coeff
        # Normalize the class weights
        class_weights = class_weights / torch.sum(class_weights) * self.num_classes

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_weights = class_weights.to(device)
        # class-aware loss
        class_aware_loss = torch.sum(class_weights * cross_entropy_loss)

        return class_aware_loss

