import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, average_precision_score
from sklearn.metrics import multilabel_confusion_matrix, classification_report

# --- Configurations ---
DATA_DIR = "dataset/house_1"
TRAIN_FILE = os.path.join(DATA_DIR, "train_data.csv")
TEST_FILE = os.path.join(DATA_DIR, "test_data.csv")
MODEL_SAVE_PATH = "models/eco_multilabel_model_e20_base.pth"

# Columns in dataset
FEATURE_COLS = [
    "powerallphases",
    "powerl1",
    "powerl2",
    "powerl3",
    "currentneutral",
    "currentl1",
    "currentl2",
    "currentl3",
    "voltagel1",
    "voltagel2",
    "voltagel3",
    "phaseanglevoltagel2l1",
    "phaseanglevoltagel3l1",
    "phaseanglecurrentvoltagel1",
    "phaseanglecurrentvoltagel2",
    "phaseanglecurrentvoltagel3",
]

TARGET_COLS = [
    "Fridge",
    "Dryer",
    "Coffee machine",
    "Kettle",
    "Washing machine",
    "PC (including router)",
    "Freezer",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Dataset Class ---
class EcoDataset(Dataset):
    def __init__(
        self, csv_file, feature_cols, target_cols, scaler=None, fit_scaler=False
    ):
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
            # nn.Sigmoid()  # sigmoid for multilabel binary probs
        )

    def forward(self, x):
        return self.net(x)


# --- Metrics helpers ---
# def calculate_metrics(y_true, y_pred, threshold=0.5):
#     y_pred_bin = (y_pred >= threshold).astype(int)
#     y_true = np.asarray(y_true).astype(int)
#     print("y_true shape:", y_true.shape, y_true.dtype)
#     print("y_pred_bin shape:", y_pred_bin.shape, y_pred_bin.dtype)
#     f1 = f1_score(y_true, y_pred_bin, average="samples", zero_division=0)
#     recall = recall_score(y_true, y_pred_bin, average="samples", zero_division=0)
#     precision = precision_score(y_true, y_pred_bin, average="samples", zero_division=0)
#     accuracy = accuracy_score(y_true, y_pred_bin)

#     # mAP calculation: mean average precision over labels (use precision/recall curve)
#     # Here we do a simplified version: average of precisions
#     mAP = precision  # simplified proxy

#     return f1, recall, precision, accuracy, mAP, y_pred_bin
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate metrics for multilabel classification
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted probabilities (n_samples, n_labels)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        f1, recall, precision, accuracy, mAP, y_pred_bin
    """
    # Convert to numpy arrays and ensure proper format
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_pred_bin = (y_pred >= threshold).astype(np.int32)
    
    # Ensure 2D shape
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred_bin.ndim == 1:
        y_pred_bin = y_pred_bin.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # print("y_true shape:", y_true.shape, y_true.dtype)
    # print("y_pred_bin shape:", y_pred_bin.shape, y_pred_bin.dtype)
    # print("y_true unique values:", np.unique(y_true))
    # print("y_pred_bin unique values:", np.unique(y_pred_bin))
    
    # Calculate metrics manually to avoid sklearn format detection issues
    n_samples, n_labels = y_true.shape
    
    # Initialize metric arrays
    f1_scores = []
    recall_scores = []
    precision_scores = []
    
    # Calculate metrics for each label separately
    for label_idx in range(n_labels):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred_bin[:, label_idx]
        
        # Calculate TP, FP, FN for this label
        tp = np.sum((y_true_label == 1) & (y_pred_label == 1))
        fp = np.sum((y_true_label == 0) & (y_pred_label == 1))
        fn = np.sum((y_true_label == 1) & (y_pred_label == 0))
        
        # Calculate precision, recall, f1 for this label
        precision_label = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_label = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_label = 2 * (precision_label * recall_label) / (precision_label + recall_label) if (precision_label + recall_label) > 0 else 0
        
        f1_scores.append(f1_label)
        recall_scores.append(recall_label)
        precision_scores.append(precision_label)
    
    # Average across labels (macro averaging)
    f1 = np.mean(f1_scores)
    recall = np.mean(recall_scores)
    precision = np.mean(precision_scores)
    
    # Calculate subset accuracy (exact match accuracy for multilabel)
    accuracy = np.mean(np.all(y_true == y_pred_bin, axis=1))
    
    # Calculate mAP manually
    mAP_scores = []
    for label_idx in range(n_labels):
        y_true_label = y_true[:, label_idx]
        y_pred_label = y_pred[:, label_idx]
        
        # Sort by prediction confidence
        sorted_indices = np.argsort(y_pred_label)[::-1]
        y_true_sorted = y_true_label[sorted_indices]
        
        # Calculate average precision for this label
        if np.sum(y_true_sorted) == 0:
            ap = 0.0
        else:
            precisions = []
            for k in range(1, len(y_true_sorted) + 1):
                precision_at_k = np.sum(y_true_sorted[:k]) / k
                if y_true_sorted[k-1] == 1:  # Only consider positions where true label is 1
                    precisions.append(precision_at_k)
            ap = np.mean(precisions) if precisions else 0.0
        
        mAP_scores.append(ap)
    
    mAP = np.mean(mAP_scores)
    
    return f1, recall, precision, accuracy, mAP, y_pred_bin


def calculate_metrics_detailed(y_true, y_pred, threshold=0.5, label_names=None):
    """
    Calculate detailed metrics for multilabel classification including per-label metrics
    
    Args:
        y_true: Ground truth labels (n_samples, n_labels)
        y_pred: Predicted probabilities (n_samples, n_labels)
        threshold: Classification threshold (default: 0.5)
        label_names: List of label names for reporting (optional)
    
    Returns:
        Dictionary with various metrics
    """
    y_pred_bin = (y_pred >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    
    # Calculate metrics with different averaging strategies
    metrics = {
        'f1_samples': f1_score(y_true, y_pred_bin, average="samples", zero_division=0),
        'f1_macro': f1_score(y_true, y_pred_bin, average="macro", zero_division=0),
        'f1_micro': f1_score(y_true, y_pred_bin, average="micro", zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred_bin, average="weighted", zero_division=0),
        
        'recall_samples': recall_score(y_true, y_pred_bin, average="samples", zero_division=0),
        'recall_macro': recall_score(y_true, y_pred_bin, average="macro", zero_division=0),
        'recall_micro': recall_score(y_true, y_pred_bin, average="micro", zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred_bin, average="weighted", zero_division=0),
        
        'precision_samples': precision_score(y_true, y_pred_bin, average="samples", zero_division=0),
        'precision_macro': precision_score(y_true, y_pred_bin, average="macro", zero_division=0),
        'precision_micro': precision_score(y_true, y_pred_bin, average="micro", zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred_bin, average="weighted", zero_division=0),
        
        'accuracy': accuracy_score(y_true, y_pred_bin),
        'mAP': average_precision_score(y_true, y_pred, average="samples"),
    }
    
    # Per-label metrics
    f1_per_label = f1_score(y_true, y_pred_bin, average=None, zero_division=0)
    recall_per_label = recall_score(y_true, y_pred_bin, average=None, zero_division=0)
    precision_per_label = precision_score(y_true, y_pred_bin, average=None, zero_division=0)
    
    if label_names:
        for i, label in enumerate(label_names):
            metrics[f'f1_{label}'] = f1_per_label[i]
            metrics[f'recall_{label}'] = recall_per_label[i]
            metrics[f'precision_{label}'] = precision_per_label[i]
    
    return metrics, y_pred_bin

# --- Train & Test functions ---
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0
    all_targets = []
    all_preds = []
    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        raw_outputs = model(x)
        outputs = torch.sigmoid(raw_outputs)
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


# def test_epoch(model, dataloader, criterion):
#     model.eval()
#     running_loss = 0
#     all_targets = []
#     all_preds = []
#     with torch.no_grad():
#         for x, y in tqdm(dataloader, desc="Testing", leave=False):
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             raw_outputs = model(x)
#             outputs = torch.sigmoid(raw_outputs)

#             loss = criterion(outputs, y)
#             running_loss += loss.item() * x.size(0)

#             all_targets.append(y.detach().cpu().numpy())
#             all_preds.append(outputs.detach().cpu().numpy())

#     epoch_loss = running_loss / len(dataloader.dataset)
#     all_targets = np.vstack(all_targets)
#     all_preds = np.vstack(all_preds)
#     f1, recall, precision, accuracy, mAP, y_pred_bin = calculate_metrics(all_targets, all_preds)

#     # Full classification report (per label)
#     # print("\n=== Classification Report (Test) ===")
#     # print(classification_report(all_targets, y_pred_bin, target_names=TARGET_COLS, zero_division=0))

#     print("\n=== Classification Report (Test) ===")
#     print(classification_report(all_targets.astype(int), y_pred_bin.astype(int), target_names=TARGET_COLS, zero_division=0))

#     return epoch_loss, f1, recall, precision, accuracy, mAP


def test_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Testing", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            raw_outputs = model(x)
            outputs = torch.sigmoid(raw_outputs)

            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)

            all_targets.append(y.detach().cpu().numpy())
            all_preds.append(outputs.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    all_targets = np.vstack(all_targets)
    all_preds = np.vstack(all_preds)
    f1, recall, precision, accuracy, mAP, y_pred_bin = calculate_metrics(
        all_targets, all_preds
    )

    true_bin = all_targets.astype(int)
    pred_bin = y_pred_bin.astype(int)

    assert true_bin.shape == pred_bin.shape
    print("\n=== Classification Report (Test) ===")
    print(
        classification_report(
            true_bin, pred_bin, target_names=TARGET_COLS, zero_division=0
        )
    )

    return epoch_loss, f1, recall, precision, accuracy, mAP


# --- Main Pipeline ---
def main():
    # Load train dataset with scaler fitting
    train_dataset = EcoDataset(
        TRAIN_FILE, FEATURE_COLS, TARGET_COLS, scaler=None, fit_scaler=True
    )
    scaler = train_dataset.scaler

    # Load test dataset using the same scaler
    test_dataset = EcoDataset(
        TEST_FILE, FEATURE_COLS, TARGET_COLS, scaler=scaler, fit_scaler=False
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = MultiLabelClassifier(
        input_dim=len(FEATURE_COLS), output_dim=len(TARGET_COLS)
    ).to(DEVICE)
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    # To keep track for plots
    history = {
        "train_loss": [],
        "train_f1": [],
        "train_recall": [],
        "train_precision": [],
        "train_acc": [],
        "train_map": [],
        "test_loss": [],
        "test_f1": [],
        "test_recall": [],
        "test_precision": [],
        "test_acc": [],
        "test_map": [],
    }

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_f1, train_recall, train_precision, train_acc, train_map = (
            train_epoch(model, train_loader, criterion, optimizer)
        )
        test_loss, test_f1, test_recall, test_precision, test_acc, test_map = (
            test_epoch(model, test_loader, criterion)
        )

        print(
            f"Train Loss: {train_loss:.4f} | F1: {train_f1:.4f} | Recall: {train_recall:.4f} | Precision: {train_precision:.4f} | Acc: {train_acc:.4f} | mAP: {train_map:.4f}"
        )
        print(
            f"Test  Loss: {test_loss:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f} | Precision: {test_precision:.4f} | Acc: {test_acc:.4f} | mAP: {test_map:.4f}"
        )

        # Save history
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["train_recall"].append(train_recall)
        history["train_precision"].append(train_precision)
        history["train_acc"].append(train_acc)
        history["train_map"].append(train_map)

        history["test_loss"].append(test_loss)
        history["test_f1"].append(test_f1)
        history["test_recall"].append(test_recall)
        history["test_precision"].append(test_precision)
        history["test_acc"].append(test_acc)
        history["test_map"].append(test_map)

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Plotting function
    def plot_metric(history, metric_name, MODEL_SAVE_PATH):
        plt.figure(figsize=(8, 5))
        plt.plot(history[f"train_{metric_name}"], label=f"Train {metric_name}")
        plt.plot(history[f"test_{metric_name}"], label=f"Test {metric_name}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} over epochs")
        plt.legend()
        plt.grid(True)
        make_dir_if_not_exists(MODEL_SAVE_PATH[:-4])
        plt.savefig(f"{MODEL_SAVE_PATH[:-4]}/{metric_name}.png", bbox_inches='tight')

    # Plot all metrics
    for metric in ["loss", "f1", "recall", "precision", "acc", "map"]:
        plot_metric(history, metric, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
