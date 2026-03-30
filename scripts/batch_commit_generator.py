import argparse
import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Load environment variables from .env file
load_dotenv()

# Global OpenAI client, initialized lazily
client = None

def get_client():
    """Initializes and returns the OpenAI client, reusing it if already created."""
    global client
    if client is None:
        # api_key = os.getenv("OPENAI_DEEPSEEK_API_KEY")
        # base_url = os.getenv("OPENAI_DEEPSEEK_BASE_URL")
        api_key = os.getenv("OPENAI_QWEN_GPT_API_KEY")
        base_url = os.getenv("OPENAI_QWEN_GPT_BASE_URL")
        if not api_key or not base_url:
            raise ValueError(
                "OPENAI_API_KEY and/or OPENAI_BASE_URL environment variables not set. "
                "Please create a .env file and set them, or set them as environment variables."
            )
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def call_llm(messages: list, model_name: str):
    """Calls the LLM with exponential backoff retry logic."""
    try:
        aclient = get_client()
        response = await aclient.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,  # Commit messages are relatively short
            temperature=0
        )
        return response
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise

def make_progress(note: str = "processing"):
    """Creates a rich progress bar."""
    return Progress(
        TextColumn(f"{note} • [progress.percentage]{{task.percentage:>3.0f}}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )

async def run_task(
    task: dict,
    p: Progress,
    task_id,
    output_file: str,
    semaphore: asyncio.Semaphore,
    model_name: str,
):
    """Runs a single task, including the LLM call and writing the result."""
    response_obj = None

    # Handle different task formats
    messages = task.get("messages")
    if not messages and "message" in task:
        # Construct messages from single prompt
        messages = [{"role": "user", "content": task["message"]}]
    
    if not messages:
        # Skip if no valid message found
        p.update(task_id, advance=1)
        return None

    async with semaphore:
        try:
            response_obj = await call_llm(messages, model_name)
        except Exception as e:
            # print(f"Failed to get response for task_id {task.get('task_id', 'unknown')}: {e}")
            response_obj = None

    p.update(task_id, advance=1)

    if response_obj is None:
        result = {
            "task_id": task.get("task_id"),
            "model": model_name,
            "response": None,  # Indicate failure
        }
    else:
        response_content = response_obj.choices[0].message.content
        result = {
            "task_id": task.get("task_id"),
            "model": model_name,
            "response": response_content,
        }

    # Append the result to the output file
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")

    return result

async def run_tasks_file(input_file: str, output_file: str, concurrency: int, model_name: str):
    """Runs tasks from a single file."""
    # Ensure the output file is cleared before starting
    with open(output_file, 'w', encoding='utf-8') as f:
        pass # Clears the file

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = []
            for line in f:
                if line.strip():
                    try:
                        tasks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"Loaded {len(tasks)} tasks from {input_file}.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    
    if not tasks:
        print(f"No tasks found in {input_file}")
        return

    semaphore = asyncio.Semaphore(concurrency)
    with make_progress(f"Generating {os.path.basename(output_file)}") as p:
        progress_task_id = p.add_task("Querying LLM", total=len(tasks))
        coros = [
            run_task(t, p, progress_task_id, output_file, semaphore, model_name)
            for t in tasks
        ]
        await asyncio.gather(*coros)

async def main(input_path: str, output_path: str, concurrency: int, model_name: str):
    """Main function to load tasks and orchestrate the run."""
    
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        files = [f for f in os.listdir(input_path) if f.endswith(".jsonl")]
        print(f"Found {len(files)} JSONL files in {input_path} to process.")
        
        for filename in files:
            input_file = os.path.join(input_path, filename)
            
            # Construct output filename
            # E.g. tasks_BM25... -> predictions_BM25...
            if filename.startswith("tasks_"):
                out_filename = filename.replace("tasks_", "predictions_", 1)
            else:
                out_filename = f"predictions_{filename}"
                
            output_file = os.path.join(output_path, out_filename)
            
            print(f"Processing {input_file} -> {output_file}")
            await run_tasks_file(input_file, output_file, concurrency, model_name)
            
    else:
        # Single file mode
        await run_tasks_file(input_path, output_path, concurrency, model_name)
        
    print(f"\nAll processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process commit message generation tasks from a JSONL file or directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSONL file or directory containing tasks.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSONL file or directory to save responses.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of concurrent tasks (default: 20).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3.2-20251201-128k",
        help="The name of the model to use (e.g., deepseek-v3.2-20251201-128k).",
    )
    args = parser.parse_args()

    asyncio.run(main(args.input, args.output, args.workers, args.model))
