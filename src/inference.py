import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from torch import nn
import os

MODEL_PATH = "models/eco_multilabel_model_e100.pth"  
TEST_FILE = "dataset/house_1/test_data.csv"
THRESHOLD = 0.5 

FEATURE_COLS = [
    "powerallphases", "powerl1", "powerl2", "powerl3",
    "currentneutral", "currentl1", "currentl2", "currentl3",
    "voltagel1", "voltagel2", "voltagel3",
    "phaseanglevoltagel2l1", "phaseanglevoltagel3l1",
    "phaseanglecurrentvoltagel1", "phaseanglecurrentvoltagel2", "phaseanglecurrentvoltagel3"
]

TARGET_COLS = [
    "Fridge", "Dryer", "Coffee machine", "Kettle",
    "Washing machine", "PC (including router)", "Freezer"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        )

    def forward(self, x):
        return self.net(x)

class EcoDataset(Dataset):
    def __init__(self, csv_file, feature_cols, scaler):
        self.df = pd.read_csv(csv_file)
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.features = scaler.transform(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        return x

def run_inference():
    print("[INFO] Loading test data and trained model...")

    df_test = pd.read_csv(TEST_FILE)
    X = df_test[FEATURE_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 

    dataset = EcoDataset(TEST_FILE, FEATURE_COLS, scaler)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = MultiLabelClassifier(input_dim=len(FEATURE_COLS), output_dim=len(TARGET_COLS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for x in dataloader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).float()
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds)

    df_preds = pd.DataFrame(preds, columns=TARGET_COLS)
    df_probs = pd.DataFrame(probs, columns=[col + "_prob" for col in TARGET_COLS])

    output_df = pd.concat([df_preds, df_probs], axis=1)
    output_df.to_csv("eco_test_predictions.csv", index=False)
    print("[INFO] Saved predictions to eco_test_predictions.csv")

    return output_df


if __name__ == "__main__":
    preds = run_inference()
    print(preds.head())
