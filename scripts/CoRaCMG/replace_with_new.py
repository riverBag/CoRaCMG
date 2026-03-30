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
        print("[INFO] Encoding diff...")
        embedding = model.encode([diff], normalize_embeddings=True)  # Built-in L2 normalization

        # Check whether the embedding contains NaN or Inf
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            print("[ERROR] NaN or Inf detected in the embedding.")
            return None

        print(f"[INFO] Embedding shape: {embedding.shape}")
        return embedding

    except Exception as e:
        print(f"[ERROR] Error during encoding: {e}")
        return None

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

# Iterate through all embeddings and check vectors containing NaN or Inf
for idx, emb in enumerate(embeddings):
    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
        print(f"[INFO] Re-generating embedding for item at index {idx}...")

        # Regenerate the embedding via process_single_diff using the preloaded model
        new_embedding = process_single_diff(raw_items[idx], model)  # Generate a new embedding from the original entry

        if new_embedding is not None:
            # Replace the original embedding
            embeddings[idx] = new_embedding[0]  # New embedding is 2D; use the first element

# Save updated data back to file
with open(output_file_path, 'wb') as f:
    pickle.dump(jina_data, f)

# Output result
print(f"[INFO] Updated embeddings for items with NaN or Inf.")
