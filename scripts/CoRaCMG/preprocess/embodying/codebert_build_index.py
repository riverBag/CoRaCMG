# -*- coding: utf-8 -*-
"""
CodeBERT diff embedding index builder (with L2 normalization)
- Only uses `diff` field from jsonl
- Model: microsoft/codebert-base
- Output: pickle index for semantic retrieval
"""

import json
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "/root/autodl-tmp/codebert-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_INDEX = "../../resource/codebert_diff_index.pkl"


# -----------------------------
# Load JSONL
# -----------------------------
def load_diffs(jsonl_path):
    diffs = []
    raw_items = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error at line {line_no}: {e}")

            diff = item.get("diff", "")
            diffs.append(diff)
            raw_items.append(item)

    return diffs, raw_items


# -----------------------------
# CodeBERT Encoder
# -----------------------------
class CodeBERTEncoder:
    def __init__(self, model_name, device):
        # AutoTokenizer + AutoModel can automatically match Roberta/Bert checkpoints
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state  # (B, L, H)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        masked_hidden = last_hidden * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        lengths = attention_mask.sum(dim=1)

        embeddings = sum_hidden / lengths  # (B, H)
        # L2 normalization
        norms = np.linalg.norm(embeddings.cpu().numpy(), axis=1, keepdims=True)
        embeddings = embeddings.cpu().numpy() / norms

        return embeddings


# -----------------------------
# Main
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python codebert_build_index.py <full.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    print(f"[INFO] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("[INFO] CUDA is not available. Using CPU.")

    print("[INFO] Loading diffs...")
    diffs, raw_items = load_diffs(jsonl_path)
    print(f"[INFO] Loaded {len(diffs)} entries")

    print("[INFO] Loading CodeBERT model...")
    encoder = CodeBERTEncoder(MODEL_NAME, DEVICE)

    print("[INFO] Encoding diffs...")
    all_embeddings = []
    for i in tqdm(range(0, len(diffs), BATCH_SIZE)):
        batch = diffs[i:i + BATCH_SIZE]
        emb = encoder.encode(batch)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)  # (N, H)

    print("[INFO] Saving embedding index...")
    with open(OUTPUT_INDEX, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "raw_items": raw_items,
                "model": MODEL_NAME
            },
            f
        )

    print(f"[INFO] Index saved to {OUTPUT_INDEX}")
    print(f"[INFO] Embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
