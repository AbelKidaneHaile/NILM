import torch
import torch.nn as nn
import torch.nn.functional as F

class CUSTOM_MULTILABEL_CLASSIFIER(nn.Module):
    def __init__(self, input_dim=16, output_dim=7):
        super(EcoNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, output_dim) 
        )

    def forward(self, x):
        return self.net(x)
