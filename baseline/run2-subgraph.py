#!/usr/bin/env python3

import yaml
import sys
from pathlib import Path
from draw import get_field_dependency_adjacency, analyze_field_dependencies
# Add parent directory to Python path to find dabstep_benchmark module
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import time
import ast
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
import pandas as pd
from dabstep_benchmark.utils import evaluate
from smolagents.utils import console
from utils import TqdmLoggingHandler
from constants import REPO_ID
from tqdm import tqdm
from prompts import (
    reasoning_llm_system_prompt,
    reasoning_llm_task_prompt,
    chat_llm_task_prompt,
    chat_llm_system_prompt
)
from utils import (
    is_reasoning_llm,
    create_code_agent_with_chat_llm,
    create_code_agent_with_reasoning_llm,
    get_tasks_to_run,
    append_answer,
    append_console_output,
    download_context
)

logging.basicConfig(level=logging.WARNING, handlers=[TqdmLoggingHandler()])
logger = logging.getLogger(__name__)


def load_referred_tasks_mapping():
    """Load the mapping of task_id to referred task IDs from clustered_llm_based.csv"""
    csv_path = Path(__file__).resolve().parent.parent / "data" / "task_cluster" / "clustered_llm_based.csv"
    df = pd.read_csv(csv_path)
    mapping = {}
    
    for _, row in df.iterrows():
        task_id = str(row['task_id'])
        referred_str = str(row['referred'])
        
        try:
            if referred_str and referred_str != 'nan':
                referred_ids = ast.literal_eval(referred_str)
                mapping[task_id] = [str(ref_id) for ref_id in referred_ids] if isinstance(referred_ids, list) else []
            else:
                mapping[task_id] = []
        except:
            mapping[task_id] = []
    
    return mapping


def load_encode_data():
    """Load all encode data from JSON files"""
    encode_data_dir = Path(__file__).resolve().parent.parent / "data" / "encode" / "encode_data"
    encode_data = {}
    
    for json_file in encode_data_dir.glob("*.json"):
        filename = json_file.stem  # Get filename without extension
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                encode_data[filename] = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load encode data from {json_file}: {e}")
            encode_data[filename] = {}
    
    return encode_data


def load_subgraph():
    """
    Load subgraph adjacency dictionaries for each task from infer_fields.csv
    
    Returns:
        dict: Mapping of task_id to field information with encode data and related fields
            Format: {field_name: {encode_data..., 'adjecent_fields': [list_of_connected_fields]}, ...}
            Example: {
                'payments.merchant': {
                    'data_type': 'string',
                    'description': 'Name of the merchant processing the transaction',
                    'data_sample': ['Crossfit_Hanna', 'Belles_cookbook_store'],
                    'adjecent_fields': ['merchant_data.merchant', 'payments.aci']
                }
            }
    """
    csv_path = Path(__file__).resolve().parent.parent / "data" / "task_cluster" / "infer_fields.csv"
    df = pd.read_csv(csv_path)
    
    # Load encode data once
    encode_data = load_encode_data()
    
    subgraph_mapping = {}
    
    for _, row in df.iterrows():
        task_id = str(row['task_id'])
        
        # Parse input fields
        input_fields_str = str(row['input_fields'])
        if input_fields_str and input_fields_str != 'nan':
            input_fields = [field.strip() for field in input_fields_str.split(',')]
        else:
            input_fields = []
        
        # Parse output fields  
        output_fields_str = str(row['output_fields'])
        if output_fields_str and output_fields_str != 'nan':
            output_fields = [field.strip() for field in output_fields_str.split(',')]
        else:
            output_fields = []
        
        # Get adjacency dictionary and enrich with encode data
        if input_fields:  # Only process if we have input fields
            try:
                # get_field_dependency_adjacency returns a dict where:
                # - Keys are field names (e.g., 'payments.merchant')
                # - Values are lists of connected field names in the subgraph
                # - Only includes nodes that are reachable from input_fields to output_fields
                adjacency_dict = get_field_dependency_adjacency(input_fields, output_fields)
                
                # Create flattened structure with encode data and related fields
                flattened_dict = {}
                for field_name, connections in adjacency_dict.items():
                    # Parse filename and field from field_name (format: filename.field)
                    if '.' in field_name:
                        filename, field = field_name.split('.', 1)
                        
                        # Start with encode data for this field
                        field_info = {}
                        if filename in encode_data and field in encode_data[filename]:
                            field_info = encode_data[filename][field].copy()
                        
                        # Add related fields from adjacency
                        field_info['adjecent_fields'] = connections
                        
                        flattened_dict[field_name] = field_info
                    else:
                        # If field_name doesn't follow filename.field pattern, keep as is
                        flattened_dict[field_name] = {
                            'adjecent_fields': connections
                        }
                
                subgraph_mapping[task_id] = flattened_dict
            except Exception as e:
                logger.warning(f"Failed to get adjacency for task {task_id}: {e}")
                subgraph_mapping[task_id] = {}  # Empty dict if adjacency computation fails
        else:
            subgraph_mapping[task_id] = {}  # Empty dict if no input fields

    return subgraph_mapping


def get_referred_trajectories(referred_ids: list[str]) -> str:
    """Get the trajectory content for the referred task IDs"""
    if not referred_ids:
        return ""
    
    trajectories_dir = Path(__file__).resolve().parent.parent / "data" / "trajectories"
    trajectories_content = []
    
    for ref_id in referred_ids:
        trajectory_file = trajectories_dir / f"{ref_id}.md"
        if trajectory_file.exists():
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                content = f.read()
            trajectories_content.append(f"=== Example Trajectory {ref_id} ===\n{content}\n")
    
    return "\n".join(trajectories_content) if trajectories_content else ""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--model-id", type=str, default="openai/o3-mini")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--tasks-ids", type=int, nargs="+", default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--use-azure-auth", action="store_true", help="Use Azure managed identity authentication instead of API key")
    parser.add_argument("--split", type=str, default="default", choices=["default", "dev"])
    parser.add_argument("--timestamp", type=str, default=None)
    return parser.parse_args()


def run_single_task(
        task: dict,
        model_id: str,
        api_base: str,
        api_key: str,
        use_azure_auth: bool,
        ctx_path: str,
        base_filename: Path,
        is_dev_data: bool,
        max_steps: int,
        referred_mapping: dict,
        subgraph_mapping: dict
):
    task_id = str(task["task_id"])
    referred_ids = referred_mapping.get(task_id, [])
    referred_examples = get_referred_trajectories(referred_ids)
    
    # Get subgraph information for this task
    task_subgraph = subgraph_mapping.get(task_id, {})
    subgraph_info = ""
    if task_subgraph:
        subgraph_info = f"\n\nField Dependencies and Schema:\n{json.dumps(task_subgraph, indent=2)}\n"

    if is_reasoning_llm(model_id):
        prompt = reasoning_llm_task_prompt.format(
            question=task["question"],
            guidelines=task["guidelines"],
            referred_examples=f"\n\nReferred Examples:\n{referred_examples}" if referred_examples else "",
            subgraph=subgraph_info
        )
        agent = create_code_agent_with_reasoning_llm(model_id, api_base, api_key, max_steps, ctx_path, use_azure_auth)
        prompt = agent.system_prompt + "\n" + prompt
        agent.system_prompt = ""
    else:
        prompt = chat_llm_task_prompt.format(
            ctx_path=ctx_path,
            question=task["question"],
            guidelines=task["guidelines"],
            referred_examples=f"\n\nReferred Examples:\n{referred_examples}" if referred_examples else "",
            subgraph=subgraph_info
        )
        agent = create_code_agent_with_chat_llm(model_id, api_base, api_key, max_steps, use_azure_auth)

    with console.capture() as capture:
        answer = agent.run(prompt)

    logger.warning(f"Task id: {task['task_id']}\tQuestion: {task['question']} Answer: {answer}\n{'=' * 50}")

    answer_dict = {"task_id": str(task["task_id"]), "agent_answer": str(answer)}
    answers_file = base_filename / "answers.jsonl"
    logs_file = base_filename / "logs.txt"

    if is_dev_data:
        scores = evaluate(agent_answers=pd.DataFrame([answer_dict]), tasks_with_gt=pd.DataFrame([task]))
        entry = {**answer_dict, "answer": task["answer"], "score": scores[0]["score"], "level": scores[0]["level"]}
        append_answer(entry, answers_file)
    else:
        append_answer(answer_dict, answers_file)
    append_console_output(capture.get(), logs_file)


def main():
    args = parse_args()
    logger.warning(f"Starting run with arguments: {args}")

    ctx_path = download_context(str(Path().resolve()))
    referred_mapping = load_referred_tasks_mapping()
    subgraph_mapping = load_subgraph()

    runs_dir = Path().resolve() / "runs2-subgraph"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.time() if not args.timestamp else args.timestamp
    base_filename = runs_dir / f"{args.model_id.replace('/', '_').replace('.', '_')}/{args.split}/{int(timestamp)}"

    # save config
    os.makedirs(base_filename, exist_ok=True)
    with open(base_filename / "config.yaml", "w", encoding="utf-8") as f:
        if is_reasoning_llm(args.model_id):
            args.system_prompt = reasoning_llm_system_prompt
        else:
            args.system_prompt = chat_llm_system_prompt
        args_dict = vars(args)
        yaml.dump(args_dict, f, default_flow_style=False)

    # Load dataset with user-chosen split
    # data = datasets.load_dataset(REPO_ID, name="tasks", split=args.split, download_mode='force_redownload')
    data = datasets.load_dataset(REPO_ID, name="tasks", split=args.split)

    if args.max_tasks >= 0 and args.tasks_ids is not None:
        logger.error(f"Can not provide {args.max_tasks=} and {args.tasks_ids=} at the same time")
    total = len(data) if args.max_tasks < 0 else min(len(data), args.max_tasks)

    tasks_to_run = get_tasks_to_run(data, total, base_filename, args.tasks_ids)
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(
                run_single_task,
               task,
               args.model_id,
               args.api_base,
               args.api_key,
               args.use_azure_auth,
               ctx_path,
               base_filename,
               (args.split == "dev"),
               args.max_steps,
               referred_mapping,
               subgraph_mapping
            )
            for task in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    logger.warning("All tasks processed.")


if __name__ == "__main__":
    main()
