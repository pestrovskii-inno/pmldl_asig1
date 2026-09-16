import os
import sys
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_def import MnistCNN

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed")
MODEL_DIR = os.path.join(REPO_ROOT, "models")

EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_split(name):
    data = np.load(os.path.join(DATA_DIR, name))
    images = data["images"].astype(np.float32) / 255.0
    images = images.reshape(len(images), 1, 28, 28)
    labels = data["labels"].astype(np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def train():
    # Feature engineering - normalize pixels to 0-1 and reshape to channels first
    train_x, train_y = load_split("train.npz")
    test_x, test_y = load_split("test.npz")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=BATCH_SIZE)

    model = MnistCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(xb)
        print(f"epoch {epoch + 1} loss {total_loss / seen:.4f} accuracy {correct / seen:.4f} time {time.time() - t0:.1f}s")
        mlflow.log_metrics({"train_loss": total_loss / seen, "train_accuracy": correct / seen}, step=epoch + 1)

    # Evaluate on the test set
    model.eval()
    correct, seen = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            correct += (out.argmax(1) == yb).sum().item()
            seen += len(xb)
    test_acc = correct / seen
    print(f"test accuracy {test_acc:.4f}")
    mlflow.log_metric("test_accuracy", test_acc)

    # Package the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "mnist_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"model saved to {model_path}")

    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="model",
        code_paths=[os.path.join(os.path.dirname(__file__), "model_def.py")],
        input_example=torch.rand(1, 1, 28, 28),
        serialization_format="pickle",
    )


def tracking_uri():
    db = os.path.join(REPO_ROOT, "mlflow.db")
    if os.path.isabs(db):
        return "sqlite:////" + db.lstrip("/")
    return "sqlite:///" + db


def main():
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("MNIST_CNN")
    with mlflow.start_run(run_name="mnist_cnn"):
        mlflow.log_params({"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LR, "model": "MnistCNN"})
        train()


if __name__ == "__main__":
    main()
