# CoRaCMG Replication Package

Replication code for the paper "CoRaCMG: Contextual Retrieval-Augmented Framework for Commit Message Generation".

![CoRaCMG Overview](overview.png)

## Overview

This repository currently contains:

- Retrieval/indexing pipelines based on BM25, Jina embedding, and CodeBERT embedding.
- Prompt/task generation scripts for commit message generation.
- Batch LLM generation script.
- Evaluation script with BLEU, ROUGE-L, METEOR, and CIDEr.

## Updated Project Structure

```text
CoRaCMG/
├── ApacheCM/
│   └── downloadLink.txt
├── environment.yml
├── overview.png
├── readme.md
└── scripts/
  ├── batch_commit_generator.py
  ├── eval.py
  ├── prompt.py
  ├── task_generator.py
  ├── metric/
  │   ├── __init__.py
  │   ├── cider.py
  │   └── cider_scorer.py
  └── CoRaCMG/
    ├── check_embeddings.py
    ├── replace_with_new.py
    ├── embedding/
    │   ├── __init__.py
    │   ├── codebert_build_index.py
    │   ├── jina_build_diff_index.py
    │   └── jina_build_diff_index_fix_nan.py
    ├── preprocess/
    │   ├── bm25_indexing.py
    │   ├── query_generate.py
    │   └── embodying/
    │       ├── codebert_build_index.py
    │       └──  jina_build_diff_index.py
    └── retriever/
      ├── __init__.py
      ├── bm25_codebert.py
      ├── bm25_jina.py
      ├── check_single_embeddings.py
      └── retrieve_query.py
```

Notes:

- `requirements.txt` is removed. Dependency management is now unified in `environment.yml`.
- IDE metadata under `scripts/CoRaCMG/.idea/` is intentionally omitted from the structure above.

## Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate CoRaCMG
```

Main dependencies are now managed in `environment.yml`, including:

- python=3.12
- numpy
- pandas
- torch
- transformers
- scikit-learn
- tqdm
- jina
- sentence-transformers
- fire
- evaluate
- datasets

## Data Preparation

The repository currently only includes `ApacheCM/downloadLink.txt` as a data pointer.

Before running the pipeline, prepare required JSONL files (for example `full.jsonl`, `test.jsonl`) and place them in paths expected by your scripts.

Several scripts use fixed default paths such as `../resource/...`; create those folders/files as needed.

## Typical Workflow

### 1. Build BM25 index

```bash
cd scripts/CoRaCMG/preprocess
python bm25_indexing.py /path/to/full.jsonl
```

### 2. Build dense embedding index

Jina index:

```bash
cd scripts/CoRaCMG/embedding
python jina_build_diff_index.py /path/to/full.jsonl
```

CodeBERT index:

```bash
cd scripts/CoRaCMG/embedding
python codebert_build_index.py /path/to/full.jsonl
```

### 3. Run hybrid retrieval

Jina + BM25:

```bash
cd scripts/CoRaCMG/retriever
python bm25_jina.py
```

CodeBERT + BM25:

```bash
cd scripts/CoRaCMG/retriever
python bm25_codebert.py
```

### 4. Build LLM task files

From retrieval result files:

```bash
python scripts/task_generator.py --input_path /path/to/results.jsonl --output_path /path/to/tasks.jsonl --mode similar
```

Or directly from dataset with optional retrieval database:

```bash
python scripts/prompt.py --dataset_path /path/to/test.jsonl --tasks_path /path/to/tasks.jsonl --prompt_type similar --database_path /path/to/full.jsonl
```

### 5. Generate commit messages in batch

```bash
python scripts/batch_commit_generator.py --input /path/to/tasks.jsonl --output /path/to/predictions.jsonl --workers 20 --model deepseek-v3.2-20251201-128k
```

### 6. Evaluate

```bash
python scripts/eval.py --result_jsonl /path/to/results_for_eval.jsonl
```

## Input and Output Formats

### Retrieval result JSONL (example fields)

- `query-sha`
- `query-diff`
- `retrieve-sha`
- `retrieve-diff`
- `retrieve-message`
- `score`

### Task JSONL (for generation)

- `task_id`
- `message` or `messages`

### Evaluation result JSONL (required by `scripts/eval.py`)

Each line should include:

- `task_id`
- `model`
- `label`
- `pred`

## Important Notes

- Some scripts use hard-coded default paths and assume execution from specific working directories. If your data layout differs, update those paths before running.
- `scripts/prompt.py` imports `commit.*` modules. Ensure your environment includes that package/module layout if you use this script directly.


