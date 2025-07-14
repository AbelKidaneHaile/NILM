import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    multilabel_confusion_matrix,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SmartMeterDataset(Dataset):
    """Custom Dataset for Smart Meter Data"""

    def __init__(self, csv_path, scaler=None, is_train=True):
        self.data = pd.read_csv(csv_path)

        # Define feature columns and target columns
        self.feature_cols = [
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

        self.target_cols = [
            "Fridge",
            "Dryer",
            "Coffee machine",
            "Kettle",
            "Washing machine",
            "PC (including router)",
            "Freezer",
        ]

        # Extract features and targets
        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.targets = self.data[self.target_cols].values.astype(np.float32)

        # Scale features
        if is_train:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            if scaler is not None:
                self.scaler = scaler
                self.features = self.scaler.transform(self.features)
            else:
                raise ValueError("Scaler must be provided for test dataset")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])

    def get_scaler(self):
        return self.scaler


class SmartMeterClassifier(nn.Module):
    """Neural Network for Multi-label Classification"""

    def __init__(
        self, input_dim, hidden_dims=[512, 256, 128], output_dim=7, dropout=0.3
    ):
        super(SmartMeterClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.network(x)
        return self.sigmoid(x)


class MetricsTracker:
    """Track training and validation metrics"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_recall_scores = []
        self.val_recall_scores = []
        self.train_map_scores = []
        self.val_map_scores = []

    def update(
        self,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        train_f1,
        val_f1,
        train_recall,
        val_recall,
        train_map,
        val_map,
    ):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.train_f1_scores.append(train_f1)
        self.val_f1_scores.append(val_f1)
        self.train_recall_scores.append(train_recall)
        self.val_recall_scores.append(val_recall)
        self.train_map_scores.append(train_map)
        self.val_map_scores.append(val_map)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate various metrics for multilabel classification"""
    y_pred_binary = (y_pred > threshold).astype(int)

    # Accuracy (exact match)
    accuracy = accuracy_score(y_true, y_pred_binary)

    # F1 Score (macro average)
    f1 = f1_score(y_true, y_pred_binary, average="macro", zero_division=0)

    # Recall (macro average)
    recall = recall_score(y_true, y_pred_binary, average="macro", zero_division=0)

    # Mean Average Precision
    map_score = average_precision_score(y_true, y_pred, average="macro")

    return accuracy, f1, recall, map_score


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with tqdm(dataloader, desc="Training") as pbar:
        for features, targets in pbar:
            features, targets = features.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_predictions.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

            pbar.set_postfix(f"loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy, f1, recall, map_score = calculate_metrics(all_targets, all_predictions)

    return avg_loss, accuracy, f1, recall, map_score


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as pbar:
            for features, targets in pbar:
                features, targets = features.to(device), targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(targets.detach().cpu().numpy())

                pbar.set_postfix(f"loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy, f1, recall, map_score = calculate_metrics(all_targets, all_predictions)

    return avg_loss, accuracy, f1, recall, map_score, all_predictions, all_targets


def plot_training_metrics(metrics_tracker, save_dir="./"):
    """Plot all training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Training and Validation Metrics", fontsize=16, fontweight="bold")

    epochs = range(1, len(metrics_tracker.train_losses) + 1)

    # Loss
    axes[0, 0].plot(
        epochs, metrics_tracker.train_losses, "b-", label="Training Loss", linewidth=2
    )
    axes[0, 0].plot(
        epochs, metrics_tracker.val_losses, "r-", label="Validation Loss", linewidth=2
    )
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(
        epochs,
        metrics_tracker.train_accuracies,
        "b-",
        label="Training Accuracy",
        linewidth=2,
    )
    axes[0, 1].plot(
        epochs,
        metrics_tracker.val_accuracies,
        "r-",
        label="Validation Accuracy",
        linewidth=2,
    )
    axes[0, 1].set_title("Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[0, 2].plot(
        epochs, metrics_tracker.train_f1_scores, "b-", label="Training F1", linewidth=2
    )
    axes[0, 2].plot(
        epochs, metrics_tracker.val_f1_scores, "r-", label="Validation F1", linewidth=2
    )
    axes[0, 2].set_title("F1 Score", fontweight="bold")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("F1 Score")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Recall
    axes[1, 0].plot(
        epochs,
        metrics_tracker.train_recall_scores,
        "b-",
        label="Training Recall",
        linewidth=2,
    )
    axes[1, 0].plot(
        epochs,
        metrics_tracker.val_recall_scores,
        "r-",
        label="Validation Recall",
        linewidth=2,
    )
    axes[1, 0].set_title("Recall", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # mAP
    axes[1, 1].plot(
        epochs,
        metrics_tracker.train_map_scores,
        "b-",
        label="Training mAP",
        linewidth=2,
    )
    axes[1, 1].plot(
        epochs,
        metrics_tracker.val_map_scores,
        "r-",
        label="Validation mAP",
        linewidth=2,
    )
    axes[1, 1].set_title("Mean Average Precision (mAP)", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("mAP")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Combined metrics
    axes[1, 2].plot(
        epochs, metrics_tracker.val_accuracies, "r-", label="Accuracy", linewidth=2
    )
    axes[1, 2].plot(
        epochs, metrics_tracker.val_f1_scores, "g-", label="F1 Score", linewidth=2
    )
    axes[1, 2].plot(
        epochs, metrics_tracker.val_recall_scores, "b-", label="Recall", linewidth=2
    )
    axes[1, 2].plot(
        epochs, metrics_tracker.val_map_scores, "m-", label="mAP", linewidth=2
    )
    axes[1, 2].set_title("Validation Metrics Comparison", fontweight="bold")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_metrics.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def generate_classification_report(y_true, y_pred, target_names, save_dir="./"):
    """Generate detailed classification report"""
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Generate classification report
    report = classification_report(
        y_true,
        y_pred_binary,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Print report
    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 80)
    print(
        classification_report(
            y_true, y_pred_binary, target_names=target_names, zero_division=0
        )
    )

    # Save report to file
    with open(os.path.join(save_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Create confusion matrix for each class
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred_binary)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, (cm, appliance) in enumerate(zip(conf_matrices, target_names)):
        if i < len(axes):
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                ax=axes[i],
                cmap="Blues",
                xticklabels=["Off", "On"],
                yticklabels=["Off", "On"],
            )
            axes[i].set_title(f"{appliance}", fontweight="bold")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

    # Hide the last subplot if not needed
    if len(target_names) < len(axes):
        axes[-1].set_visible(False)

    plt.suptitle(
        "Confusion Matrices for Each Appliance", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "confusion_matrices.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    return report


def main():
    """Main training and evaluation pipeline"""
 
    # Configuration
    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "hidden_dims": [512, 256, 128],
        "dropout": 0.3,
        "patience": 15,  # Early stopping patience
        "data_dir": "house_1_chunks",
        "save_dir": "models",
    }

    # Create save directory
    os.makedirs(config["save_dir"], exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = SmartMeterDataset(
        os.path.join(config["data_dir"], "train.csv"), is_train=True
    )

    test_dataset = SmartMeterDataset(
        os.path.join(config["data_dir"], "test.csv"),
        scaler=train_dataset.get_scaler(),
        is_train=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Input features: {len(train_dataset.feature_cols)}")
    print(f"Output classes: {len(train_dataset.target_cols)}")

    # Initialize model
    model = SmartMeterClassifier(
        input_dim=len(train_dataset.feature_cols),
        hidden_dims=config["hidden_dims"],
        output_dim=len(train_dataset.target_cols),
        dropout=config["dropout"],
    ).to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, verbose=True
    )

    # Metrics tracker
    metrics_tracker = MetricsTracker()

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting training...")
    print("=" * 80)

    # Training loop
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)

        # Train
        train_loss, train_acc, train_f1, train_recall, train_map = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_recall, val_map, _, _ = validate_epoch(
            model, test_loader, criterion, device
        )

        # Update metrics
        metrics_tracker.update(
            train_loss,
            val_loss,
            train_acc,
            val_acc,
            train_f1,
            val_f1,
            train_recall,
            val_recall,
            train_map,
            val_map,
        )

        # Print epoch results
        print(
            f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
            f"Recall: {train_recall:.4f}, mAP: {train_map:.4f}"
        )
        print(
            f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
            f"Recall: {val_recall:.4f}, mAP: {val_map:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "scaler": train_dataset.get_scaler(),
                },
                os.path.join(config["save_dir"], "best_model.pth"),
            )
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

    print("\nTraining completed!")
    print("=" * 80)

    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(config["save_dir"], "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final evaluation
    print("\nFinal Evaluation on Test Set:")
    print("-" * 50)

    _, test_acc, test_f1, test_recall, test_map, test_predictions, test_targets = (
        validate_epoch(model, test_loader, criterion, device)
    )

    print(
        f"Test Results - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
        f"Recall: {test_recall:.4f}, mAP: {test_map:.4f}"
    )

    # Generate plots and reports
    print("\nGenerating visualizations and reports...")
    plot_training_metrics(metrics_tracker, config["save_dir"])

    # Classification report
    target_names = train_dataset.target_cols
    report = generate_classification_report(
        test_targets, test_predictions, target_names, config["save_dir"]
    )

    # Save final model (different from best model checkpoint)
    torch.save(model.state_dict(), os.path.join(config["save_dir"], "final_model.pth"))

    # Save training history
    history = {
        "train_losses": metrics_tracker.train_losses,
        "val_losses": metrics_tracker.val_losses,
        "train_accuracies": metrics_tracker.train_accuracies,
        "val_accuracies": metrics_tracker.val_accuracies,
        "train_f1_scores": metrics_tracker.train_f1_scores,
        "val_f1_scores": metrics_tracker.val_f1_scores,
        "train_recall_scores": metrics_tracker.train_recall_scores,
        "val_recall_scores": metrics_tracker.val_recall_scores,
        "train_map_scores": metrics_tracker.train_map_scores,
        "val_map_scores": metrics_tracker.val_map_scores,
    }

    with open(os.path.join(config["save_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nAll files saved to: {config['save_dir']}")
    print("Files created:")
    print("- best_model.pth (best model with full checkpoint)")
    print("- final_model.pth (final model state dict)")
    print("- training_metrics.png (training plots)")
    print("- confusion_matrices.png (per-class confusion matrices)")
    print("- classification_report.json (detailed metrics)")
    print("- training_history.json (metrics history)")

    return model, report, history


if __name__ == "__main__":
    model, report, history = main()
