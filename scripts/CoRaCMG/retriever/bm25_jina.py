# -*- coding: utf-8 -*-

import json
import pickle
import sys
import os
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from preprocess.bm25_indexing import tokenize_diff, BM25


# =========================
# Utility functions
# =========================
def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_v, max_v = scores.min(), scores.max()
    if min_v == max_v:
        return np.zeros_like(scores)
    scores = (scores - min_v) / (max_v - min_v)
    if np.any(np.isnan(scores)):
        raise ValueError("NaN detected in normalized scores")
    return scores


def save_results(results, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Main logic
# =========================
def main():
    RESOURCE_DIR = "../resource"
    BM25_INDEX_PATH = os.path.join(RESOURCE_DIR, "bm25_diff_index.pkl")
    JINA_INDEX_PATH = os.path.join(RESOURCE_DIR, "jina_diff_index_fixed.pkl")
    QUERY_JSONL_PATH = os.path.join(RESOURCE_DIR, "apachecm/test.jsonl")

    WEIGHTS = [(5, 5)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading Jina encoder on {device}")
    jina_encoder = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
        device=device
    )
    jina_encoder.max_seq_length = 4096

    print("[INFO] Loading indices")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25_model: BM25 = bm25_data["bm25"]
        bm25_items = bm25_data["raw_items"]

    with open(JINA_INDEX_PATH, "rb") as f:
        jina_data = pickle.load(f)
        jina_embeddings = jina_data["embeddings"]

    # repo -> indices
    repo_map = defaultdict(list)
    for idx, item in enumerate(bm25_items):
        repo_map[item["repo"]].append(idx)

    # load queries
    queries = []
    with open(QUERY_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # =========================
    # Precompute BM25 + Dense
    # =========================
    query_cache = {}

    print("[INFO] Precomputing BM25 & Dense scores")
    for q in tqdm(queries):
        q_sha = q["commit_sha"]
        q_repo = q["repo"]
        q_diff = q.get("diff", "")

        # Filter by repo and exclude self
        candidates = [
            idx for idx in repo_map.get(q_repo, [])
            if bm25_items[idx]["commit_sha"] != q_sha
        ]
        if not candidates:
            continue

        q_tokens = tokenize_diff(q_diff)

        # BM25
        bm25_scores = np.array([
            bm25_model.score(q_tokens, idx)
            for idx in candidates
        ])
        bm25_scores = min_max_normalize(bm25_scores)

        # Dense (Jina) - cosine similarity
        q_emb = jina_encoder.encode(q_diff, normalize_embeddings=True)
        cand_embs = jina_embeddings[candidates]

        # L2 normalization (to ensure cosine similarity)
        q_emb_norm = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        cand_embs_norm = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10)

        dense_scores = np.dot(cand_embs_norm, q_emb_norm)  # [-1,1]
        dense_scores = (dense_scores + 1.0) / 2.0          # [0,1]
        dense_scores = min_max_normalize(dense_scores)

        query_cache[q_sha] = {
            "indices": candidates,
            "bm25": bm25_scores,
            "dense": dense_scores
        }

    # =========================
    # Fusion & Top-1
    # =========================
    for w_bm25, w_dense in WEIGHTS:
        results = []

        for q in tqdm(queries, desc=f"BM25:{w_bm25} Dense:{w_dense}"):
            q_sha = q["commit_sha"]
            q_diff = q.get("diff", "")

            cache = query_cache.get(q_sha)
            if cache is None:
                continue

            final_scores = (
                w_bm25 * cache["bm25"] +
                w_dense * cache["dense"]
            )

            best_pos = int(np.argmax(final_scores))
            best_idx = cache["indices"][best_pos]
            best_item = bm25_items[best_idx]

            results.append({
                "query-sha": q_sha,
                "query-diff": q_diff,
                "retrieve-sha": best_item["commit_sha"],
                "retrieve-diff": best_item["diff"],
                "retrieve-message": best_item.get("message", ""),
                "bm25-score": float(cache["bm25"][best_pos]),
                "dense-score": float(cache["dense"][best_pos]),
                "score": float(final_scores[best_pos]),
                "weight": f"BM25:{w_bm25} Dense:{w_dense}"
            })

        out_path = os.path.join(
            RESOURCE_DIR,
            f"results_w_BM25_dense_{w_bm25}_{w_dense}_Jina.jsonl"
        )
        save_results(results, out_path)
        print(f"[INFO] Saved {out_path}")


if __name__ == "__main__":
    main()
