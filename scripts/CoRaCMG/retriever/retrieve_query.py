import json 
import pickle
import sys
import os
import re
import math
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
# Add the project root to sys.path to allow imports from sibling directories 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embedding.codebert_build_index import CodeBERTEncoder
from preprocess.bm25_indexing import tokenize_diff, BM25

# ==========================================
# Main Retrieval Logic
# ==========================================

def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Normalize the scores using min-max normalization.
    """
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val:
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def convert_numpy_types(result):
    """
    Convert any numpy types (like np.float32, np.int32) to standard Python types (like float, int).
    """
    return {key: (value.item() if isinstance(value, (np.float32, np.int32, np.float64)) else value) for key, value in result.items()}

def save_results(results, output_file):
    """
    Save the results to a JSONL file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            result = convert_numpy_types(result)  # Ensure numpy types are converted
            f.write(json.dumps(result) + "\n")

def main():
    # Paths
    RESOURCE_DIR = "../resource"
    BM25_INDEX_PATH = os.path.join(RESOURCE_DIR, "bm25_diff_index.pkl")
    CODEBERT_INDEX_PATH = os.path.join(RESOURCE_DIR, "codebert_diff_index.pkl")
    JINA_INDEX_PATH = os.path.join(RESOURCE_DIR, "jina_diff_index.pkl")
    QUERY_JSONL_PATH = os.path.join(RESOURCE_DIR, "query.jsonl")

    # Weights configuration (BM25 : Dense)
    WEIGHTS = [
        (0, 10),
        (3, 7),
        (5, 5),
        (7, 3),
        (10, 0)
    ]



    # Initialize Models for Query Encoding
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing CodeBERT model on {device}...")
    cb_encoder = CodeBERTEncoder("microsoft/codebert-base", device)

    print(f"[INFO] Initializing Jina model on {device}...")
    jina_encoder = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)

    print("[INFO] Loading indices...")

    # Load BM25 Index
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_data = pickle.load(f)
        bm25_model: BM25 = bm25_data["bm25"]
        bm25_items = bm25_data["raw_items"]

    # Load CodeBERT Index
    with open(CODEBERT_INDEX_PATH, "rb") as f:
        cb_data = pickle.load(f)
        cb_embeddings = cb_data["embeddings"]

    # Load Jina Index
    with open(JINA_INDEX_PATH, "rb") as f:
        jina_data = pickle.load(f)
        jina_embeddings = jina_data["embeddings"]

    # Verify alignment
    assert len(bm25_items) == len(cb_embeddings) == len(jina_embeddings)

    # Pre-compute repo mapping
    repo_map = defaultdict(list)
    for idx, item in enumerate(bm25_items):
        repo_map[item.get("repo")].append(idx)

    print(f"[INFO] Loaded indices. Total documents: {len(bm25_items)}")

    # Load Queries
    queries = []
    with open(QUERY_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    print(f"[INFO] Loaded {len(queries)} queries.")

    # Initialize the empty candidate counter
    empty_candidate_counter = 0

    # Iterate over queries
    results_log = []

    for q_idx, query in enumerate(tqdm(queries, desc="Processing Queries")):
        q_repo = query.get("repo")
        q_sha = query.get("commit_sha")
        q_diff = query.get("diff", "")

        # 1. Identify Candidates (Same Repo)
        candidate_indices = repo_map.get(q_repo, [])
        if not candidate_indices:
            print("=" * 40)
            continue
        print(f"[INFO] Query {q_idx}: Repo={q_repo}, SHA={q_sha}, Candidates={len(candidate_indices)}")
        # 2. Compute BM25 Scores for Candidates
        q_tokens = tokenize_diff(q_diff)
        bm25_scores = []
        for idx in candidate_indices:
            bm25_scores.append(bm25_model.score(q_tokens, idx))
        bm25_scores = np.array(bm25_scores)

        # Normalize BM25 Scores (Min-Max)
        bm25_scores_norm = min_max_normalize(bm25_scores)

        # 3. Compute Dense Scores & Hybrid Fusion
        cb_q_emb = cb_encoder.encode(q_diff)  # Shape (1, H)
        cb_cand_embs = cb_embeddings[candidate_indices]  # (M, H)
        cb_scores = np.dot(cb_cand_embs, cb_q_emb.T).flatten()
        cb_scores_norm = min_max_normalize(cb_scores)

        MAX_LENGTH = 4096  # Set maximum length
        tokenizer = jina_encoder.tokenizer
        inputs = tokenizer([q_diff], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        jina_q_emb = jina_encoder.encode([inputs['input_ids']], normalize_embeddings=True)
        jina_cand_embs = jina_embeddings[candidate_indices]
        jina_scores = np.dot(jina_cand_embs, jina_q_emb.T).flatten()
        jina_scores_norm = min_max_normalize(jina_scores)

        # 4. Apply Weights and Find Best Match
        def process_fusion(dense_name, dense_scores_norm):
            nonlocal empty_candidate_counter
            print(f"[INFO] Query Model={dense_name}")
            for (w_bm25, w_dense) in WEIGHTS:
                final_scores = (w_bm25 * bm25_scores_norm) + (w_dense * dense_scores_norm)

                # Create (score, idx) pairs
                scored_candidates = []
                for i, score in enumerate(final_scores):
                    global_idx = candidate_indices[i]
                    item = bm25_items[global_idx]

                    # Apply filters:
                    # 1. Ensure the same repo
                    if item.get("repo") != q_repo:
                        continue

                    # 2. Exclude self (query itself)
                    if item.get("commit_sha") == q_sha:
                        continue

                    scored_candidates.append((score, item))

                # Check if candidate list is empty
                if not scored_candidates:
                    empty_candidate_counter += 1
                print(f"[INFO] Query {q_idx}: Repo={q_repo}, SHA={q_sha}, Model={dense_name}, len(scored_candidates)={len(scored_candidates)}")
                # Sort and pick top 1 result for each query and model
                if scored_candidates:
                    scored_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_item = scored_candidates[0]

                    # Prepare result
                    result = {
                        "query-diff": q_diff,
                        "query-sha": q_sha,
                        "retrieve-diff": best_item["diff"],
                        "retrieve-message": best_item.get("message", ""),
                        "retrieve-sha": best_item["commit_sha"],
                        "bm25-score": bm25_scores_norm[scored_candidates.index((best_score, best_item))],
                        "module-score": dense_scores_norm[scored_candidates.index((best_score, best_item))],
                        "score": best_score,
                        "model": dense_name  # Add model name to differentiate between CodeBERT and Jina
                    }

                    # Append result to log for this query, only once
                    if not any(r["query-sha"] == q_sha for r in results_log):
                        results_log.append(result)
                else:
                    print(f"Query: {q_sha[:7]} | Model: {dense_name} | W(BM25:Dense)={w_bm25}:{w_dense} | No match found")

        # Run for CodeBERT
        process_fusion("CodeBERT", cb_scores_norm)

        # Run for Jina
        process_fusion("Jina", jina_scores_norm)

        print("-" * 40)

    # After processing all queries, save the results
    print(f"[INFO] Total processed queries: {len(results_log)}")
    print(f"[INFO] Total skipped entries (empty candidates): {empty_candidate_counter}")
    print("[INFO] Saving results...")

    # Save results for each weight combination
    for (w_bm25, w_dense) in WEIGHTS:
        # Filter results by model type (CodeBERT or Jina)
        # codebert_results = [r for r in results_log if r["bm25-score"] > 0 and r["module-score"] > 0 and r["model"] == "CodeBERT"]
        # jina_results = [r for r in results_log if r["bm25-score"] > 0 and r["module-score"] > 0 and r["model"] == "Jina"]
        codebert_results = [r for r in results_log if r["model"] == "CodeBERT"]
        jina_results = [r for r in results_log if r["model"] == "Jina"]

        # Save CodeBERT results
        codebert_output_file = os.path.join(RESOURCE_DIR, f"results_w_BM25_dense_{w_bm25}_{w_dense}_CodeBERT.jsonl")
        save_results(codebert_results, codebert_output_file)
        print(f"[INFO] Saved CodeBERT results to {codebert_output_file}")

        # Save Jina results
        jina_output_file = os.path.join(RESOURCE_DIR, f"results_w_BM25_dense_{w_bm25}_{w_dense}_Jina.jsonl")
        save_results(jina_results, jina_output_file)
        print(f"[INFO] Saved Jina results to {jina_output_file}")


if __name__ == "__main__":
    main()
