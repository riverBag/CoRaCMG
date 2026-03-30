# -*- coding: utf-8 -*-
"""
Build diff embedding index using local jina-embeddings-v2-base-code
Only uses `diff` field from full.jsonl
"""

import json
import sys
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


#
# Config
#
MODEL_PATH = "jinaai/jina-embeddings-v2-base-code"
OUTPUT_PATH = "../../resource/jina_diff_index.pkl"
BATCH_SIZE = 2          # Local model is usually more stable with a smaller batch size
MAX_SEQ_LENGTH = 4096     # Strongly recommended to truncate to avoid token explosion

#
# Load JSONL
#
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


#
# Main (Index Builder)
#
def main():
    if len(sys.argv) < 2:
        print("Usage: python jina_build_diff_index.py <full.jsonl>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    print("[INFO] Loading diffs...")
    diffs, raw_items = load_diffs(jsonl_path)
    print(f"[INFO] Loaded {len(diffs)} diffs")

    print(f"[INFO] Loading local embedding model from: {MODEL_PATH}")

    model = SentenceTransformer(
        MODEL_PATH,
        trust_remote_code=True,
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    print("[INFO] Encoding diffs (local model)...")
    all_vectors = []

    for i in tqdm(range(0, len(diffs), BATCH_SIZE)):
        batch = diffs[i : i + BATCH_SIZE]

        vectors = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            normalize_embeddings=True,   # Built-in L2 normalization
        )

        all_vectors.append(vectors)

    embeddings = np.vstack(all_vectors)
    print(f"[INFO] Total embeddings shape: {embeddings.shape}")

    print(f"[INFO] Saving index to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "raw_items": raw_items,
                "model_path": MODEL_PATH,
                "max_seq_length": MAX_SEQ_LENGTH,
            },
            f,
        )

    print("[INFO] Index saved successfully.")


if __name__ == "__main__":
    main()
