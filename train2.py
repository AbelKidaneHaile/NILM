import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configurations ---
DATA_DIR = "house_1_chunks"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODEL_SAVE_PATH = "eco_multilabel_model.pth"

# Columns in dataset
FEATURE_COLS = ['powerallphases', 'powerl1', 'powerl2', 'powerl3', 'currentneutral', 
                'currentl1', 'currentl2', 'currentl3', 'voltagel1', 'voltagel2', 
                'voltagel3', 'phaseanglevoltagel2l1', 'phaseanglevoltagel3l1', 
                'phaseanglecurrentvoltagel1', 'phaseanglecurrentvoltagel2', 'phaseanglecurrentvoltagel3']

TARGET_COLS = ['Fridge', 'Dryer', 'Coffee machine', 'Kettle', 'Washing machine', 'PC (including router)', 'Freezer']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class EcoDataset(Dataset):
    def __init__(self, csv_file, feature_cols, target_cols, scaler=None, fit_scaler=False):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[target_cols].values.astype(np.float32)
        
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.features = self.scaler.transform(self.features)
        else:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y

# --- Model ---
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Sigmoid()  # sigmoid for multilabel binary probs
        )
        
    def forward(self, x):
        return self.net(x)

# --- Metrics helpers ---
def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    
    f1 = f1_score(y_true, y_pred_bin, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred_bin, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred_bin)
    
    # mAP calculation: mean average precision over labels (use precision/recall curve)
    # Here we do a simplified version: average of precisions
    mAP = precision  # simplified proxy
    
    return f1, recall, precision, accuracy, mAP, y_pred_bin

# --- Train & Test functions ---
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    all_targets = []
    all_preds = []
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        all_targets.append(y.detach().cpu().numpy())
        all_preds.append(outputs.detach().cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    f1, recall, precision, accuracy, mAP, _ = calculate_metrics(all_targets, all_preds)
    
    return epoch_loss, f1, recall, precision, accuracy, mAP

def test_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Testing", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            
            all_targets.append(y.detach().cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    f1, recall, precision, accuracy, mAP, y_pred_bin = calculate_metrics(all_targets, all_preds)
    
    # Full classification report (per label)
    print("\n=== Classification Report (Test) ===")
    print(classification_report(all_targets, y_pred_bin, target_names=TARGET_COLS, zero_division=0))
    
    return epoch_loss, f1, recall, precision, accuracy, mAP

# --- Main Pipeline ---
def main():
    # Load train dataset with scaler fitting
    train_dataset = EcoDataset(TRAIN_FILE, FEATURE_COLS, TARGET_COLS, scaler=None, fit_scaler=True)
    scaler = train_dataset.scaler
    
    # Load test dataset using the same scaler
    test_dataset = EcoDataset(TEST_FILE, FEATURE_COLS, TARGET_COLS, scaler=scaler, fit_scaler=False)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    model = MultiLabelClassifier(input_dim=len(FEATURE_COLS), output_dim=len(TARGET_COLS)).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 30
    
    # To keep track for plots
    history = {'train_loss': [], 'train_f1': [], 'train_recall': [], 'train_precision': [], 'train_acc': [], 'train_map': [],
               'test_loss': [], 'test_f1': [], 'test_recall': [], 'test_precision': [], 'test_acc': [], 'test_map': []}
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_f1, train_recall, train_precision, train_acc, train_map = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_f1, test_recall, test_precision, test_acc, test_map = test_epoch(model, test_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Recall: {train_recall:.4f} | Precision: {train_precision:.4f} | Acc: {train_acc:.4f} | mAP: {train_map:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f} | Precision: {test_precision:.4f} | Acc: {test_acc:.4f} | mAP: {test_map:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_f1'].append(train_f1)
        history['train_recall'].append(train_recall)
        history['train_precision'].append(train_precision)
        history['train_acc'].append(train_acc)
        history['train_map'].append(train_map)
        
        history['test_loss'].append(test_loss)
        history['test_f1'].append(test_f1)
        history['test_recall'].append(test_recall)
        history['test_precision'].append(test_precision)
        history['test_acc'].append(test_acc)
        history['test_map'].append(test_map)
    
    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    # Plotting function
    def plot_metric(history, metric_name):
        plt.figure(figsize=(8,5))
        plt.plot(history[f'train_{metric_name}'], label=f'Train {metric_name}')
        plt.plot(history[f'test_{metric_name}'], label=f'Test {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} over epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Plot all metrics
    for metric in ['loss', 'f1', 'recall', 'precision', 'acc', 'map']:
        plot_metric(history, metric)

if __name__ == "__main__":
    main()
