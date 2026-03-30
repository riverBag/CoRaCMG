# -*- coding: utf-8 -*-
"""
Fix NaN embeddings in jina_diff_index.pkl by re-embedding entries
from jina_nan_items.jsonl and replacing by commit_sha.
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
OLD_INDEX_PATH = "../resource/jina_diff_index.pkl"
OUTPUT_PATH = "../resource/jina_diff_index_fixed.pkl"

BATCH_SIZE = 2
MAX_SEQ_LENGTH = 4096


#
# Load JSONL (NaN items)
#
def load_nan_items(jsonl_path):
    diffs = []
    items = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error at line {line_no}: {e}")

            if "commit_sha" not in item:
                raise RuntimeError(f"Missing commit_sha at line {line_no}")

            diff = item.get("diff", "")
            diffs.append(diff)
            items.append(item)

    return diffs, items


#
# Main
#
def main():
    if len(sys.argv) < 2:
        print("Usage: python jina_build_diff_index_fix_nan.py <jina_nan_items.jsonl>")
        sys.exit(1)

    nan_jsonl_path = sys.argv[1]

    # ---------- 1. Load old index ----------
    print("[INFO] Loading original jina_diff_index.pkl...")
    with open(OLD_INDEX_PATH, "rb") as f:
        index_data = pickle.load(f)

    embeddings = index_data["embeddings"]
    raw_items = index_data["raw_items"]

    emb_dim = embeddings.shape[1]

    # commit_sha -> index
    sha2idx = {}
    for idx, item in enumerate(raw_items):
        sha = item.get("commit_sha")
        if sha:
            sha2idx[sha] = idx

    # ---------- 2. Load NaN items ----------
    print("[INFO] Loading NaN items...")
    diffs, nan_items = load_nan_items(nan_jsonl_path)
    print(f"[INFO] Loaded {len(diffs)} NaN diffs")

    # ---------- 3. Load model ----------
    print(f"[INFO] Loading local embedding model from: {MODEL_PATH}")
    model = SentenceTransformer(
        MODEL_PATH,
        trust_remote_code=True,
    )
    model.max_seq_length = MAX_SEQ_LENGTH

    # ---------- 4. Re-embed ----------
    print("[INFO] Re-encoding NaN diffs...")
    new_vectors = {}

    for i in tqdm(range(0, len(diffs), BATCH_SIZE)):
        batch_diffs = diffs[i : i + BATCH_SIZE]
        batch_items = nan_items[i : i + BATCH_SIZE]

        vectors = model.encode(
            batch_diffs,
            batch_size=len(batch_diffs),
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        for item, vec in zip(batch_items, vectors):
            sha = item["commit_sha"]

            if np.isnan(vec).any():
                raise RuntimeError(f"Re-embedded vector still NaN: {sha}")

            if vec.shape[0] != emb_dim:
                raise RuntimeError("Embedding dimension mismatch")

            if sha not in sha2idx:
                raise RuntimeError(f"commit_sha not found in original index: {sha}")

            new_vectors[sha] = vec

    # ---------- 5. Replace ----------
    print("[INFO] Replacing embeddings in original index...")
    for sha, vec in new_vectors.items():
        idx = sha2idx[sha]
        embeddings[idx] = vec

    print(f"[INFO] Replaced {len(new_vectors)} embeddings")

    # ---------- 6. Save ----------
    print(f"[INFO] Saving fixed index to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(index_data, f)

    print("[INFO] Done. NaN embeddings fixed.")


if __name__ == "__main__":
    main()
