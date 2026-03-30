# -*- coding: utf-8 -*-
import pickle
import numpy as np


def check_embeddings_for_nan_inf(file_path):
    """
    Check whether NaN or Inf values exist in the embedding file.
    """
    try:
        # Load file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print("[INFO] Loaded file:", file_path)
        print("[INFO] File keys:", list(data.keys()))

        # Get embeddings
        embeddings = data.get("embeddings", None)

        if embeddings is None:
            print("[WARNING] No embeddings field found in file")
            return

        # Convert to numpy array for consistency
        embeddings_array = np.array(embeddings)

        # Better check for emptiness
        if embeddings_array.size == 0 or embeddings_array.shape[0] == 0:
            print("[WARNING] embeddings is empty")
            return

        print("[INFO] Total embedding vectors:", embeddings_array.shape[0])
        print("[INFO] Dimension per vector:", embeddings_array.shape[1] if len(embeddings_array.shape) > 1 else 1)

        # Check each embedding vector
        nan_count = 0
        inf_count = 0
        problematic_indices = []

        for idx in range(embeddings_array.shape[0]):
            emb = embeddings_array[idx]

            # Check NaN
            has_nan = np.any(np.isnan(emb))
            # Check Inf
            has_inf = np.any(np.isinf(emb))

            if has_nan:
                nan_count += 1
                problematic_indices.append(idx)
                print(f"  [WARNING] Index {idx}: contains NaN values")

            if has_inf:
                inf_count += 1
                if idx not in problematic_indices:
                    problematic_indices.append(idx)
                print(f"  [WARNING] Index {idx}: contains Inf values")

            # Optional: check zero vectors or other issues
            if np.all(emb == 0):
                print(f"  [WARNING] Index {idx}: all-zero vector")

        # Output statistics
        print("\n" + "=" * 50)
        print("[Check Result Statistics]")
        print("Total embedding vectors:", embeddings_array.shape[0])
        print("Vectors containing NaN:", nan_count)
        print("Vectors containing Inf:", inf_count)
        print("Total problematic vectors:", len(problematic_indices))

        if problematic_indices:
            print("Problematic vector indices:", problematic_indices)

            # Optional: show details of a subset of problematic vectors
            print("\n[Partial Problematic Vector Details]")
            for i in problematic_indices[:min(5, len(problematic_indices))]:  # Show only first 5
                emb = embeddings_array[i]
                nan_num = np.sum(np.isnan(emb))
                inf_num = np.sum(np.isinf(emb))
                print(f"Index {i}: shape={emb.shape}, NaN count={nan_num}, Inf count={inf_num}")
        else:
            print("[SUCCESS] No NaN or Inf values found in any embedding vector!")

        # Check embedding dimensions and statistics
        if embeddings_array.shape[0] > 0:
            print(f"\n[Embedding Information]")
            print("Embedding shape:", embeddings_array.shape)
            print(f"Value range: [{embeddings_array.min():.6f}, {embeddings_array.max():.6f}]")
            print(f"Mean: {embeddings_array.mean():.6f}")
            print(f"Std dev: {embeddings_array.std():.6f}")

            # Check whether any NaN or Inf exists overall
            has_nan_overall = np.any(np.isnan(embeddings_array))
            has_inf_overall = np.any(np.isinf(embeddings_array))
            print(f"Has NaN overall: {has_nan_overall}")
            print(f"Has Inf overall: {has_inf_overall}")

        return problematic_indices

    except Exception as e:
        print("[ERROR] Error while checking file:", e)
        import traceback
        traceback.print_exc()
        return []


def compare_files(original_path, updated_path):
    """
    Compare differences between the original file and the updated file.
    """
    print("\n" + "=" * 50)
    print("[Compare Original and Updated Files]")

    try:
        # Load original file
        with open(original_path, 'rb') as f:
            original_data = pickle.load(f)

        # Load updated file
        with open(updated_path, 'rb') as f:
            updated_data = pickle.load(f)

        original_embeddings = original_data.get("embeddings", None)
        updated_embeddings = updated_data.get("embeddings", None)

        if original_embeddings is None or updated_embeddings is None:
            print("[WARNING] One of the files does not contain an embeddings field")
            return

        # Convert to numpy arrays
        orig_array = np.array(original_embeddings)
        upd_array = np.array(updated_embeddings)

        print(f"Original file embedding count: {orig_array.shape[0]}")
        print(f"Updated file embedding count: {upd_array.shape[0]}")

        # Check whether counts are consistent
        if orig_array.shape[0] != upd_array.shape[0]:
            print("[WARNING] Embedding counts are inconsistent between the two files!")
            return

        # Count fixed vectors
        fixed_count = 0
        unchanged_count = 0
        still_problematic = 0

        for i in range(orig_array.shape[0]):
            orig_emb = orig_array[i]
            upd_emb = upd_array[i]

            orig_has_nan_inf = np.any(np.isnan(orig_emb)) or np.any(np.isinf(orig_emb))
            upd_has_nan_inf = np.any(np.isnan(upd_emb)) or np.any(np.isinf(upd_emb))

            if orig_has_nan_inf and not upd_has_nan_inf:
                fixed_count += 1
            elif not orig_has_nan_inf and not upd_has_nan_inf:
                unchanged_count += 1
            elif orig_has_nan_inf and upd_has_nan_inf:
                still_problematic += 1
                print(f"  [WARNING] Index {i}: problematic in both original and updated")

        print(f"Fixed vector count: {fixed_count}")
        print(f"Unchanged normal vector count: {unchanged_count}")
        print(f"Still problematic vector count: {still_problematic}")

        if fixed_count > 0:
            print(f"[SUCCESS] Successfully fixed {fixed_count} problematic embedding vectors!")

        if still_problematic > 0:
            print(f"[WARNING] {still_problematic} vectors are still problematic")

        # Check whether any vectors were modified unexpectedly
        if fixed_count == 0 and still_problematic == 0:
            # No vectors contain NaN/Inf, check exact equality
            if np.array_equal(orig_array, upd_array):
                print("[INFO] Two files are exactly identical")
            else:
                # Compute differences
                diff = np.abs(orig_array - upd_array)
                max_diff = diff.max()
                mean_diff = diff.mean()
                print(f"[INFO] Files differ but contain no NaN/Inf: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    except Exception as e:
        print("[ERROR] Error while comparing files:", e)


if __name__ == "__main__":
    # File paths
    original_file = './resource/jina_diff_index.pkl'  # Original file
    updated_file = './resource/updated_jina_diff_index.pkl'  # Updated file

    print("=" * 50)
    print("Checking updated file...")
    print("=" * 50)

    # Check updated file
    problematic_indices = check_embeddings_for_nan_inf(updated_file)

    # Compare original file and updated file
    compare_files(original_file, updated_file)

    # Optional: also check original file
    print("\n" + "=" * 50)
    print("Checking original file...")
    print("=" * 50)
    check_embeddings_for_nan_inf(original_file)