import json

import numpy as np
import pandas
# import optuna
import pandas as pd
from sklearn.model_selection import train_test_split


import torch
import torch.nn as nn
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

# X_train = X_train[..., np.newaxis]  # Shape: (num_samples, 50, 1)
# X_test = X_test[..., np.newaxis]    # Shape: (num_samples, 50, 1)
# y_train = y_train[..., np.newaxis]  # Shape: (num_samples, 50, 1)
# y_test = y_test[..., np.newaxis]    # Shape: (num_samples, 50, 1)



# 示例数据集
class TextDataset(Dataset):
    def __init__(self, X, y,embedder_for_sentences):
        self.X = X
        self.y = y
        self.embedder_for_sentences = embedder_for_sentences

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Sentence embedding
        sentences = self.X[idx]
        embedding = self.embedder_for_sentences.encode(sentences)
        label = self.y[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义 RNN + 分类模型
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, (hidden, _) = self.rnn(x)  # hidden shape: (num_layers, batch_size, hidden_size)
        # 使用最后一层的隐藏状态作为特征
        final_hidden = hidden[-1]  # shape: (batch_size, hidden_size)
        out = self.fc(final_hidden)
        return out

# 初始化 Sentence Transformer
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 你可以选择其他预训练模型

# 示例数据
# sentences = ["This is a good movie", "This is a bad movie", "I love this film", "I hate this film"]
# labels = [1, 0, 1, 0]  # 1: Positive, 0: Negative

# 创建数据集和数据加载器
dataset = TextDataset(X_train, y_train, embedder)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 模型超参数
input_size = 384  # SentenceTransformer 的默认输出维度
hidden_size = 128
num_layers = 1
num_classes = 2

# 初始化模型
model = RNNClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for embeddings, labels in dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        # 前向传播
        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    test_sentence = "This film is wonderful"
    test_embedding = torch.tensor(embedder.encode(test_sentence), dtype=torch.float32).unsqueeze(0).to(device)
    output = model(test_embedding.unsqueeze(1))  # 加入 seq_len 维度
    prediction = torch.argmax(output, dim=1).item()
    print(f"Prediction for '{test_sentence}': {'Positive' if prediction == 1 else 'Negative'}")
