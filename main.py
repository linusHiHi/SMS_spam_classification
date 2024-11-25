import json

import numpy as np
import pandas
# import optuna
import pandas as pd
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
from torch.sparse import softmax

from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = "./config/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
sourceDataSet = config["root"]+ config["dataset"]["dir"]+config["dataset"]["cleaned_pickle"]
tag = config["data"]["tag_name"]
message=config["data"]["text_name"]
if config["pca"]:
    dim = config["pca_dim"]
else:
    dim = config["un_pca_dim"]

df = pd.read_pickle(sourceDataSet)

X = df[message]  # Feature vectors,50 dimensions
y = df[tag]  # Labels (e.g., spam/ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Precompute embeddings for the dataset
X_train_embeddings = torch.tensor(embedder.encode(X_train.tolist()), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

# Update Dataset Class
class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# Initialize dataset and dataloader
dataset = TextDataset(X_train_embeddings, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hidden, _) = self.rnn(x)  # hidden shape: (num_layers, batch_size, hidden_size)
        out = hidden[-1]  # Take the last layer's hidden state
        out = self.fc(out)  # Output layer
        return out

# Model initialization
input_size = 384  # SentenceTransformer embedding size
hidden_size = 128
num_layers = 1
num_classes = 2

model = RNNClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        embeddings, labels = batch
        embeddings, labels = embeddings.to(device), labels.to(device)

        # Forward pass
        outputs = model(embeddings.unsqueeze(1))  # Add seq_len dimension
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Testing
from sklearn.metrics import classification_report, confusion_matrix

# Model evaluation
# Precompute embeddings for the test set
X_test_embeddings = torch.tensor(embedder.encode(X_test.tolist()), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create test dataset and dataloader
test_dataset = TextDataset(X_test_embeddings, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Model evaluation on the test set
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    for batch in test_dataloader:
        embeddings, labels = batch
        embeddings, labels = embeddings.to(device), labels.to(device)

        # Forward pass
        outputs = model(embeddings.unsqueeze(1))  # Add seq_len dimension
        _, predicted = torch.max(outputs, 1)

        # Store true and predicted labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Classification report
print("Classification Report on Test Set:")
print(classification_report(y_true, y_pred))

# Optionally, manually calculate metrics
cm = confusion_matrix(y_true, y_pred)
tp = cm[1, 1]  # True Positive
tn = cm[0, 0]  # True Negative
fp = cm[0, 1]  # False Positive
fn = cm[1, 0]  # False Negative

# Calculate Precision, Recall, F1-Score
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print the results
print(f"\nManual Calculation Metrics on Test Set:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")