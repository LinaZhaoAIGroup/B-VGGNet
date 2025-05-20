#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, TensorDataset
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
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def laplace_approximation(self, inputs, targets):
        def loss_function(weight, bias):
            inputs_reshaped = inputs.view(inputs.size(0), -1)  # Flatten the input
            print(f"Inputs reshaped size: {inputs_reshaped.size()}")  # Print the inputs_reshaped size
            print(f"Weight size: {weight.size()}")  # Print the weight size
            assert inputs_reshaped.size(1) == weight.size(1), 
            outputs = F.linear(inputs_reshaped, weight, bias)
            loss = loss_fn(outputs, targets)
            return loss

        hessian = torch.autograd.functional.hessian(loss_function, (self.weight, self.bias))
        posterior_cov = torch.inverse(hessian)
        posterior_mean = (self.weight.detach(), self.bias.detach())
        return posterior_mean, posterior_cov

 
class BayesianVGGNet(nn.Module):
    def __init__(self, n_classes, input_length, conv1_out, conv2_out, fc1_out=None):
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
        
        # Calculate the flattened_input_size
        dummy_input = torch.zeros(1, 1, input_length)
        dummy_output = self.features(dummy_input)
        flattened_input_size = dummy_output.view(-1).size(0)

        if fc1_out is None:
            fc1_out = flattened_input_size

        self.classifier = nn.Sequential(
            BayesianLinear(flattened_input_size, fc1_out),
            nn.ReLU(inplace=True),
            BayesianLinear(fc1_out, fc1_out // 2),
            nn.ReLU(inplace=True),
            BayesianLinear(fc1_out // 2, n_classes)
        )

    def _get_flattened_input_size(self, input_length, conv1_out, conv2_out):
        # Pass a dummy tensor through the feature extractor to determine the flattened size
        dummy_input = torch.zeros(1, 1, input_length)
        dummy_output = self.features(dummy_input)
        return dummy_output.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        flattened_input_size = x.size(1)  # Calculate flattened_input_size dynamically
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        return x
    

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

    
def norm_ppf(x):
    return np.sqrt(2) * erfinv(2 * x - 1)

def erfinv(x):
    a = 0.147
    term1 = np.log(1 - x**2) / 2
    term2 = np.log(1 - x**2) / x
    term3 = np.sqrt(np.pi / 2) * a
    return np.sign(x) * np.sqrt(-term1 + np.sqrt(term1**2 - term2 + term3))

def _get_flattened_input_size(self, input_length, conv1_out, conv2_out):
    # Pass a dummy tensor through the feature extractor to determine the flattened size
    dummy_input = torch.zeros(1, 1, input_length)
    dummy_output = self.features(dummy_input)
    print(f"dummy_output shape: {dummy_output.shape}")
    return dummy_output.view(-1).size(0)

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
                    label = predicted_labels[i]
                    lower_bound = np.clip(means[i, label] - z * stds[i, label], 0, 1)
                    upper_bound = np.clip(means[i, label] + z * stds[i, label], 0, 1)
                    interval = {
                        'label': int(label),
                        'predictions': predictions[:, i, :],
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
                    prediction_intervals.append(interval)
    
    return prediction_intervals
