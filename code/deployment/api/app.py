import io
import os

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "models/mnist_model.pt")


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


app = FastAPI(title="MNIST digit recognition API")

_model = None


def get_model():
    global _model
    if _model is None:
        model = MnistCNN()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _model = model
    return _model


def preprocess(image):
    gray = image.convert("L")
    if gray.size == (28, 28):
        arr = np.asarray(gray, dtype=np.float32)
        # MNIST digits are bright on a dark background so invert light images
        if arr.mean() > 127:
            arr = 255.0 - arr
        return torch.from_numpy(arr / 255.0).reshape(1, 1, 28, 28)
    arr = np.asarray(gray, dtype=np.float32)
    if arr.mean() > 127:
        arr = 255.0 - arr
    mask = arr > 30
    if not mask.any():
        return torch.zeros(1, 1, 28, 28)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    digit = Image.fromarray(arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype(np.uint8))
    scale = 20.0 / max(digit.size)
    box = digit.resize((max(1, int(digit.width * scale)), max(1, int(digit.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(box, ((28 - box.width) // 2, (28 - box.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).reshape(1, 1, 28, 28)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image")
    model = get_model()
    x = preprocess(image)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0)
    digit = int(probs.argmax().item())
    return {
        "digit": digit,
        "confidence": float(probs[digit].item()),
        "probabilities": {str(i): float(p) for i, p in enumerate(probs.tolist())},
    }
