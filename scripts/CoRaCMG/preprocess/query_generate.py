import json
import random

INPUT_FILE = "../resource/apachecm/test.jsonl"
OUTPUT_FILE = "../resource/query.jsonl"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42  # Optional: keep results reproducible

def sample_jsonl(input_file, output_file, sample_size, seed=None):
    if seed is not None:
        random.seed(seed)

    # Read all data
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    total = len(lines)
    if sample_size > total:
        raise ValueError(f"Sample size {sample_size} is larger than total data count {total}")

    # Random sampling
    sampled = random.sample(lines, sample_size)

    # Write to a new jsonl file
    with open(output_file, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Randomly sampled {sample_size} entries from {total} total entries, output written to {output_file}")

if __name__ == "__main__":
    sample_jsonl(INPUT_FILE, OUTPUT_FILE, SAMPLE_SIZE, RANDOM_SEED)
