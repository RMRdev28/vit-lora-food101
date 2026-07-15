"""Food-101 classifier demo server.

    uvicorn app:app --app-dir src --port 8000
    # open http://localhost:8000

Expects the Kaggle notebook's model_export.zip unzipped into  model/  at the
project root (adapter + processor + labels.json). The base ViT downloads from
the HF hub on first startup (~330 MB, cached afterwards).
"""

import io
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FoodClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

app = FastAPI(title="Food-101 ViT+LoRA Classifier")
state = {}


@app.on_event("startup")
def load_model():
    if not (MODEL_DIR / "adapter_config.json").exists():
        raise RuntimeError(
            f"No adapter found in {MODEL_DIR}. Run the Kaggle notebook, download "
            "model_export.zip, and unzip it into the model/ folder."
        )
    print("Loading model (downloads the base ViT from the HF hub on first run)...")
    t0 = time.perf_counter()
    state["clf"] = FoodClassifier(MODEL_DIR)
    print(f"Model ready in {time.perf_counter() - t0:.1f}s")


@app.post("/api/classify")
async def classify(file: UploadFile = File(...), tta: bool = False):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "not a readable image")

    t0 = time.perf_counter()
    preds = state["clf"].predict(img, top_k=5, tta=tta)
    return {
        "predictions": preds,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "tta": tta,
    }


@app.get("/api/stats")
def stats():
    clf = state["clf"]
    out = {
        "classes": len(clf.labels),
        "model": "ViT-B/16 + LoRA (Food-101)",
    }
    if clf.results:
        out["benchmark"] = {
            "test_top1_tta": clf.results.get("tta_top1"),
            "test_top5_tta": clf.results.get("tta_top5"),
        }
    return out


app.mount("/", StaticFiles(directory=ROOT / "ui", html=True), name="ui")
