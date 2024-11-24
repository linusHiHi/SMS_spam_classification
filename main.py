import json

import pandas as pd
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, Embedding
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

config_path = "./config/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
sourceDataSet = config["root"]+ config["dataset"]["dir"]+config["dataset"]["vectorized_csv"]
with open(sourceDataSet, "r") as f:
    df = pd.read_csv(f)

X = df.iloc[:, 1:].values  # Feature vectors,50 dimensions
y = df.iloc[:, 0].values   # Labels (e.g., spam/ham)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN model
model = Sequential([
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping for training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Classification report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))