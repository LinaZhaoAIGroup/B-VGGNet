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
    def __init__(self, in_features, out_features, p=0.5):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class BayesianVGGNet(nn.Module):
    def __init__(self, n_classes, conv1_out, conv2_out, fc1_out, dropout_p=0.5):
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
        self.fcn = nn.Sequential(
            nn.Flatten(),
            BayesianLinear(512 * 43, fc1_out, dropout_p),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            BayesianLinear(fc1_out, n_classes, dropout_p)  
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fcn(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, dataloader, optimizer):
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
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def validate_epoch(model, dataloader):
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
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def erfinv(x):
    a = 0.147
    term1 = np.log(1 - x**2) / 2
    term2 = np.log(1 - x**2) / x
    term3 = np.sqrt(np.pi / 2) * a
    return np.sign(x) * np.sqrt(-term1 + np.sqrt(term1**2 - term2 + term3))

def norm_ppf(x):
    return np.sqrt(2) * erfinv(2 * x - 1)

def compute_prediction_intervals(model, dataloader, confidence_level=0.95, num_samples=100, target_labels=[1, 15, 63, 127, 221]):
    model.train()  # 启用 Dropout
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
                    interval = {
                        'label': int(predicted_labels[i]),
                        'predictions': predictions[:, i, :],
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    prediction_intervals.append(interval)
    
    return prediction_intervals
