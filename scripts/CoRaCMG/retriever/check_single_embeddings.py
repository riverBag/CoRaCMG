import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Configuration
MODEL_PATH = "jinaai/jina-embeddings-v2-base-code"
MAX_SEQ_LENGTH = 4096  # Strongly recommended to truncate to avoid token explosion


# Function to process a single diff (now accepts model as a parameter)
def process_single_diff(diff, model):
    try:
        # Encode a single diff
        embedding = model.encode([diff], normalize_embeddings=True)  # Built-in L2 normalization

        # Check whether the embedding contains NaN or Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("[ERROR] NaN or Inf detected in the embedding.")
            return None

        return embedding

    except Exception as e:
        print(f"[ERROR] Error during encoding: {e}")
        return None


def compare_embeddings(emb1, emb2, idx, tolerance=1e-6):
    """
    Compare whether two embedding vectors are equal.

    Args:
        emb1: Original embedding vector
        emb2: Newly generated embedding vector
        idx: Index ID
        tolerance: Tolerance threshold
    """
    # Convert to numpy arrays
    emb1_array = np.array(emb1)
    emb2_array = np.array(emb2[0]) if len(emb2.shape) == 2 else np.array(emb2)

    # Check whether shapes are the same
    if emb1_array.shape != emb2_array.shape:
        print(f"[Compare {idx}] Shape mismatch: {emb1_array.shape} vs {emb2_array.shape}")
        return False

    # Compute differences
    diff = np.abs(emb1_array - emb2_array)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Check whether values are within tolerance
    is_close = np.allclose(emb1_array, emb2_array, rtol=tolerance, atol=tolerance)

    if is_close:
        print(f"[Compare {idx}] ✓ Vectors are equal (max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e})")
    else:
        print(f"[Compare {idx}] ✗ Vectors are not equal (max diff: {max_diff:.6e}, mean diff: {mean_diff:.6e})")

        # Find the dimension with the largest difference
        max_diff_idx = np.argmax(diff)
        max_diff_value = diff.flatten()[max_diff_idx]
        print(f"      Largest diff position: index {max_diff_idx}, value: {max_diff_value:.6e}")
        print(f"      Original value: {emb1_array.flatten()[max_diff_idx]:.6e}, New value: {emb2_array.flatten()[max_diff_idx]:.6e}")

    return is_close, max_diff, mean_diff


# Load Jina index file
jina_index_path = './resource/jina_diff_index.pkl'  # Replace with your file path
output_file_path = './resource/updated_jina_diff_index.pkl'  # Output file path

# Load model once
print(f"[INFO] Loading local embedding model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)
model.max_seq_length = MAX_SEQ_LENGTH

with open(jina_index_path, 'rb') as f:
    jina_data = pickle.load(f)

# Assume jina_data includes an "embeddings" key storing all embedding vectors
embeddings = jina_data.get("embeddings", [])
raw_items = jina_data.get("raw_items", [])  # Assume raw_items stores original entries matching embeddings (e.g., diff, sha)

# Limit processing to the first 1000 entries
limit = 1000
if len(embeddings) < limit:
    limit = len(embeddings)

print(f"[INFO] Processing only the first {limit} entries")

# Statistics
equal_count = 0
not_equal_count = 0
nan_inf_fixed = 0
all_max_diffs = []
all_mean_diffs = []

# Iterate through the first 1000 embeddings
for idx in range(limit):
    emb = embeddings[idx]

    # Check whether the original embedding contains NaN or Inf
    has_nan_inf = np.any(np.isnan(emb)) or np.any(np.isinf(emb))

    if has_nan_inf:
        print(f"[Process {idx}] Original embedding contains NaN or Inf, regenerating...")
        new_embedding = process_single_diff(raw_items[idx], model)

        if new_embedding is not None:
            # Compare new embedding with original embedding (original may be invalid, but diff is still informative)
            is_equal, max_diff, mean_diff = compare_embeddings(emb, new_embedding, idx)

            # Replace the original embedding
            embeddings[idx] = new_embedding[0]  # New embedding is 2D; use the first element
            nan_inf_fixed += 1

            if not is_equal:
                not_equal_count += 1
                all_max_diffs.append(max_diff)
                all_mean_diffs.append(mean_diff)
        else:
            print(f"[Process {idx}] Failed to regenerate embedding")
    else:
        # Original embedding is normal; regenerate and compare
        print(f"[Process {idx}] Regenerating normal embedding for comparison...")
        new_embedding = process_single_diff(raw_items[idx], model)

        if new_embedding is not None:
            # Compare new embedding with original embedding
            is_equal, max_diff, mean_diff = compare_embeddings(emb, new_embedding, idx)

            if is_equal:
                equal_count += 1
            else:
                not_equal_count += 1
                all_max_diffs.append(max_diff)
                all_mean_diffs.append(mean_diff)
        else:
            print(f"[Process {idx}] Failed to regenerate embedding")

# Output statistics
print("\n" + "=" * 50)
print("[Statistics]")
print(f"Total processed entries: {limit}")
print(f"Fixed NaN/Inf embeddings: {nan_inf_fixed}")
print(f"Equal vector count: {equal_count}")
print(f"Non-equal vector count: {not_equal_count}")

if not_equal_count > 0:
    print(f"\n[Difference Statistics]")
    print(f"Max difference range: [{min(all_max_diffs):.6e}, {max(all_max_diffs):.6e}]")
    print(f"Average of max differences: {np.mean(all_max_diffs):.6e}")
    print(f"Average difference: {np.mean(all_mean_diffs):.6e}")

    # Check whether there are significant differences
    significant_threshold = 1e-4
    significant_diffs = [d for d in all_max_diffs if d > significant_threshold]
    if significant_diffs:
        print(f"\n[WARNING] {len(significant_diffs)} embeddings exceed max difference threshold {significant_threshold}")
        print(f"Largest significant difference: {max(significant_diffs):.6e}")