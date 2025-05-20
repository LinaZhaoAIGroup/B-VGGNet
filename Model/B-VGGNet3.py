#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,random_split, ConcatDataset, TensorDataset
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torch.distributions as dist
import torchbnn as bnn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import seaborn as sns
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_rho = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.weight_rho, -5)
        nn.init.constant_(self.bias_rho, -5)

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        weight_prior = dist.Normal(0, 1)
        bias_prior = dist.Normal(0, 1)

        weight_posterior = dist.Normal(self.weight_mu, weight_sigma)
        bias_posterior = dist.Normal(self.bias_mu, bias_sigma)

        kl_weight = torch.distributions.kl.kl_divergence(weight_posterior, weight_prior).sum()
        kl_bias = torch.distributions.kl.kl_divergence(bias_posterior, bias_prior).sum()

        return kl_weight + kl_bias

class BayesianVGGNet(nn.Module):
    def __init__(self, n_classes, conv1_out, conv2_out, fc1_out, input_length):
        super(BayesianVGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, conv1_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv1_out, conv1_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv2_out, conv2_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(conv2_out, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened input size for the first fully connected layer
        self.flattened_input_size = self._get_flattened_input_size(input_length, conv1_out, conv2_out)

        self.classifier = nn.Sequential(
            BayesianLinear(self.flattened_input_size, fc1_out),
            nn.ReLU(),
            BayesianLinear(fc1_out, n_classes)
        )

    def _get_flattened_input_size(self, input_length, conv1_out, conv2_out):
        # Pass a dummy tensor through the feature extractor to determine the flattened size
        dummy_input = torch.zeros(1, 1, input_length)
        dummy_output = self.features(dummy_input)
        flattened_input_size = dummy_output.view(1, -1).size(1)  # 将 flattened_input_size 计算为每个样本的特征数量
        return dummy_output.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    def kl_divergence(self):
        kl = 0.0
        for module in self.classifier:
            if isinstance(module, BayesianLinear):
                kl += module.kl_divergence()
        return kl

# Training and validation functions
def train_epoch(model, dataloader, optimizer, kl_weight):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        inputs = inputs.unsqueeze(1)  # Add an extra dimension for the channel
        targets = targets.long()  # Convert targets to long data type
        
        optimizer.zero_grad()
        outputs = model(inputs)
        classification_loss = nn.CrossEntropyLoss()(outputs, targets)
        kl_div = model.kl_divergence()
        loss = classification_loss + kl_weight * kl_div
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate_epoch(model, dataloader, kl_weight):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            inputs = inputs.unsqueeze(1)  # Add an extra dimension for the channel
            targets = targets.long()  # Convert targets to long data type   
                     
            outputs = model(inputs)
            classification_loss = nn.CrossEntropyLoss()(outputs, targets)
            kl_div = model.kl_divergence()
            loss = classification_loss + kl_weight * kl_div
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def compute_prediction_intervals(model, dataloader, confidence_level=0.95, num_samples=100, target_labels=[1, 15, 63, 127, 221]):
    model.eval()
    prediction_intervals = []
    
    z = norm_ppf((1 + confidence_level) / 2)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            inputs = inputs.unsqueeze(1)  # Add an extra dimension for the channel
            
            predictions = []
            for _ in range(num_samples):
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())
            
            predictions = np.array(predictions)
            means = np.mean(predictions, axis=0)
            stds = np.std(predictions, axis=0)
            
            predicted_labels = np.argmax(means, axis=1)
            
            for i in range(len(predicted_labels)):
                if predicted_labels[i] in target_labels:
                    lower_bound = means[i] - z * stds[i]
                    upper_bound = means[i] + z * stds[i]
                    interval = {'label': int(predicted_labels[i]), 'predictions': predictions[:, i, :], 'lower_bound': lower_bound, 'upper_bound': upper_bound}
                    prediction_intervals.append(interval)
    
    return prediction_intervals


def norm_ppf(x):
    return np.sqrt(2) * erfinv(2 * x - 1)

def erfinv(x):
    a = 0.147
    term1 = np.log(1 - x ** 2) / 2
    term2 = np.log(1 - x ** 2) / a
    term3 = 4 / np.pi + term2 / 2
    return np.sign(x) * np.sqrt(np.sqrt(term3 ** 2 - term1) - term2)

