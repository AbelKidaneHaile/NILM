import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualFeatureNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualFeatureNet, self).__init__()
        
        # create richer feature space
        self.input_expansion = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # residual block-1
        self.residual_block1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128)
        )
        
        # learns which of the 128 features are most important
        self.feature_importance = nn.Sequential(
            nn.Linear(128, 128),
            nn.Sigmoid()
        )
        
        # residual block-2
        self.residual_block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64)
        )
        
        # dimension reduction
        self.skip_projection = nn.Linear(128, 64)
        
        # reduction -- (progressive)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        x = self.input_expansion(x)
        residual1 = self.residual_block1(x)
        x = F.relu(x + residual1)
        importance_weights = self.feature_importance(x)
        x = x * importance_weights  
        residual2 = self.residual_block2(x)
        x_projected = self.skip_projection(x) 
        x = x_projected + residual2 

        return self.classifier(x)