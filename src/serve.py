import io
import os
import sys
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import get_model

app = FastAPI(title="CIFAR-10 Classifier", version="1.0")

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = None
device = None


def load_model_from_checkpoint(checkpoint_path: str) -> torch.nn.Module:
    m = get_model(architecture="resnet18", num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m


@app.on_event("startup")
def startup_event():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
    if not Path(ckpt_path).exists():
        # allow running locally without a real checkpoint for health checks
        print(f"WARNING: checkpoint not found at {ckpt_path}, model not loaded", flush=True)
        return
    model = load_model_from_checkpoint(ckpt_path).to(device)
    print(f"Model loaded from {ckpt_path} on {device}", flush=True)


def preprocess(image_bytes: bytes) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img).unsqueeze(0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    contents = await image.read()
    tensor = preprocess(contents).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
    return JSONResponse({
        "label": LABELS[pred_idx],
        "class_index": pred_idx,
        "confidence": round(probs[pred_idx].item(), 4),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("serve:app", host="0.0.0.0", port=port, reload=False)
