"""Load the fine-tuned classifier: base ViT from the HF hub + local LoRA adapter.

The Kaggle export (model/ folder) contains only the adapter weights (LoRA matrices
+ classifier head, a few MB), the image processor config, and labels.json. The
86M-parameter base model is downloaded from the hub once and cached; the adapter
is merged into it at load time, so inference has zero LoRA overhead.
"""

import json
from pathlib import Path

import numpy as np
import torch


class FoodClassifier:
    def __init__(self, export_dir):
        from peft import PeftModel
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        export_dir = Path(export_dir)
        meta = json.loads((export_dir / "labels.json").read_text())
        self.labels = meta["labels"]

        self.processor = AutoImageProcessor.from_pretrained(export_dir)
        base = AutoModelForImageClassification.from_pretrained(
            meta["base_model"],
            num_labels=len(self.labels),
            id2label={i: l for i, l in enumerate(self.labels)},
            label2id={l: i for i, l in enumerate(self.labels)},
        )
        self.model = PeftModel.from_pretrained(base, export_dir).merge_and_unload()
        self.model.eval()

        self.results = None
        if (export_dir / "results.json").exists():
            self.results = json.loads((export_dir / "results.json").read_text())

    @torch.no_grad()
    def predict(self, pil_image, top_k=5, tta=False):
        """Returns [{label, prob}] sorted by probability."""
        img = pil_image.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, -1)

        if tta:
            flipped = img.transpose(0)  # PIL FLIP_LEFT_RIGHT
            inputs_f = self.processor(images=flipped, return_tensors="pt")
            probs = (probs + torch.softmax(self.model(**inputs_f).logits[0], -1)) / 2

        probs = probs.numpy()
        order = np.argsort(-probs)[:top_k]
        return [
            {"label": self.labels[i].replace("_", " "), "prob": round(float(probs[i]), 4)}
            for i in order
        ]
