import json
from tqdm import tqdm
import logging
import threading
from smolagents import CodeAgent
from custom_agent import CustomCodeAgent
from custom_litellm import LiteLLMModelWithBackOff
from huggingface_hub import hf_hub_download
from constants import REPO_ID, ADDITIONAL_AUTHORIZED_IMPORTS
from pathlib import Path
from prompts import reasoning_llm_system_prompt, chat_llm_system_prompt

append_answer_lock = threading.Lock()
append_console_output_lock = threading.Lock()

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record))

def read_only_open(*a, **kw):
    if (len(a) > 1 and isinstance(a[1], str) and a[1] != 'r') or kw.get('mode', 'r') != 'r':
        raise Exception("Only mode='r' allowed for the function open")
    return open(*a, **kw)

def download_context(base_dir: str) -> str:
    ctx_files = [
        "data/context/acquirer_countries.csv",
        "data/context/payments.csv",
        "data/context/merchant_category_codes.csv",
        "data/context/fees.json",
        "data/context/merchant_data.json",
        "data/context/manual.md",
        "data/context/payments-readme.md"
    ]
    for f in ctx_files:
        hf_hub_download(REPO_ID, repo_type="dataset", filename=f, local_dir=base_dir, force_download=True)

    root_dir = Path(__file__).resolve().parent.parent
    full_path = Path(base_dir) / Path(ctx_files[0]).parent
    relative_path = full_path.relative_to(root_dir)
    return str(relative_path)

def is_reasoning_llm(model_id: str) -> bool:
    reasoning_llm_list = [
        "openai/o1",
        "openai/o3",
        "openai/o3-mini",
        "o3",
        "o3-mini",
        "deepseek/deepseek-reasoner",
        "gpt5"
    ]
    return model_id in reasoning_llm_list

def get_tasks_to_run(data, total: int, base_filename: Path, tasks_ids: list[int]):
    import json
    f = base_filename.parent / f"{base_filename.stem}_answers.jsonl"
    done = set()
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            done = {json.loads(line)["task_id"] for line in fh if line.strip()}

    tasks = []
    for i in range(total):
        task_id = int(data[i]["task_id"])
        if task_id not in done:
            if tasks_ids is not None:
                if task_id in tasks_ids:
                    tasks.append(data[i])
            else:
                tasks.append(data[i])
    return tasks


def append_answer(entry: dict, jsonl_file: Path) -> None:
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")


def append_console_output(captured_text: str, txt_file: Path) -> None:
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    with append_console_output_lock, open(txt_file, "a", encoding="utf-8") as fp:
        fp.write(captured_text + "\n")

def create_code_agent_with_reasoning_llm(model_id: str, api_base=None, api_key=None, max_steps=10, ctx_path=None, use_azure_auth=False):
    agent = CustomCodeAgent(
        system_prompt=reasoning_llm_system_prompt,
        tools=[],
        model=LiteLLMModelWithBackOff(
            model_id=model_id, api_base=api_base, api_key=api_key, use_azure_auth=use_azure_auth),
        additional_authorized_imports=ADDITIONAL_AUTHORIZED_IMPORTS,
        max_steps=max_steps,
        verbosity_level=3,
    )
    agent.python_executor.static_tools.update({"open": read_only_open})

    agent.system_prompt = agent.system_prompt.format(ctx_path=ctx_path)
    return agent

def create_code_agent_with_chat_llm(model_id: str, api_base=None, api_key=None, max_steps=10, use_azure_auth=False):
    agent = CodeAgent(
        system_prompt=chat_llm_system_prompt,
        tools=[],
        model=LiteLLMModelWithBackOff(model_id=model_id, api_base=api_base, api_key=api_key, use_azure_auth=use_azure_auth),
        additional_authorized_imports=ADDITIONAL_AUTHORIZED_IMPORTS,
        max_steps=max_steps,
        verbosity_level=3,
    )

    agent.python_executor.static_tools.update({"open": read_only_open})
    return agent
