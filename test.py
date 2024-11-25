import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

config_path = "./config/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
sourceDataSet = config["root"]+ config["dataset"]["dir"]+config["dataset"]["cleaned_pickle"]

if config["pca"]:
    dim = config["pca_dim"]
else:
    dim = config["un_pca_dim"]

df = pd.read_pickle(sourceDataSet)

a = df["Message"]
b = df.iloc[0]
c = df["Message"][0]

X = df["Message"]  # Feature vectors,50 dimensions
y = df["Label"]   # Labels (e.g., spam/ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train[..., np.newaxis]
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = embedder.encode(X_train[5])
print(embeddings)