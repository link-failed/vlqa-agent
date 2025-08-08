#!/usr/bin/env python3

import yaml
import sys
from pathlib import Path
# Add parent directory to Python path to find dabstep_benchmark module
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os
import time
import ast
import random
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
    """Return a function that returns a random referred task ID from the specified list"""
    referred_pool = [5, 49, 1273, 1305, 1464, 1681, 1753, 1871, 2697]
    
    def get_random_referred_id():
        return [str(id) for id in random.sample(referred_pool, 2)]
    
    return get_random_referred_id


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
        get_random_referred_id: callable
):
    task_id = str(task["task_id"])
    referred_ids = get_random_referred_id()
    referred_examples = get_referred_trajectories(referred_ids)

    if is_reasoning_llm(model_id):
        prompt = reasoning_llm_task_prompt.format(
            question=task["question"],
            guidelines=task["guidelines"],
            referred_examples=f"\n\nReferred Examples:\n{referred_examples}" if referred_examples else ""
        )
        agent = create_code_agent_with_reasoning_llm(model_id, api_base, api_key, max_steps, ctx_path, use_azure_auth)
        prompt = agent.system_prompt + "\n" + prompt
        agent.system_prompt = ""
    else:
        prompt = chat_llm_task_prompt.format(
            ctx_path=ctx_path,
            question=task["question"],
            guidelines=task["guidelines"],
            referred_examples=f"\n\nReferred Examples:\n{referred_examples}" if referred_examples else ""
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
    get_random_referred_id = load_referred_tasks_mapping()

    runs_dir = Path().resolve() / "runs3"
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
    data = datasets.load_dataset(REPO_ID, name="tasks", split=args.split, download_mode='force_redownload')

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
               get_random_referred_id
            )
            for task in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing tasks"):
            f.result()

    logger.warning("All tasks processed.")


if __name__ == "__main__":
    main()