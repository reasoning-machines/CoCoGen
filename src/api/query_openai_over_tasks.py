"""
Given a prompt file and path to a task file with the following fields:

1. input_prompt_code: the code used to prompt codex
2. reference_code: expected completed code
3. reference_graph: expected graph

Runs inference over codex for each input_prompt_code, and adds the following fields to the output file:

4. generated_code: generated code
5. generated_graph: generated graph

The file can contain other metadata, but the fields above are required.
"""
from datetime import datetime
import shutil
import time
import openai
import pandas as pd
from tqdm import tqdm
import logging
import os

from converters.get_converter import ConverterFactory
from converters.graph_code_converter import GraphPythonConverter
from codexapi.openai_api_wrapper import OpenaiAPIWrapper
from src.prompting.constants import END

logging.basicConfig(level=logging.INFO)


def run(task_file_path: str,
        num_tasks: int,
        output_file_path: str,
        prompt_path: str,
        keep_writing_output: bool,
        engine: str,
        max_tokens:int, 
        max_requests_per_min: int):
    tasks = pd.read_json(task_file_path, orient='records', lines=True)
    converter = ConverterFactory.get_converter(args.job_type)
    if num_tasks != -1:
        tasks = tasks.head(num_tasks)

    fixed_prompt_text = read_prompt(prompt_path)

    results = []

    cache = load_cache(output_file_path)

    num_requests = 0
    time_begin = time.time()

    for task_idx, task in tqdm(tasks.iterrows(), total=len(tasks)):

        is_success = False
        for cut_prompt_examples in [None, 1, 2, 3, 4, 5, 6]:
            try:
                num_requests += 1

                request_per_minute = maintain_request_per_minute(
                    num_requests=num_requests, time_begin=time_begin, max_requests_per_min=max_requests_per_min, task_idx=task_idx)
                logging.info("\n")
                logging.info(
                    f"Task {task_idx} > request/minute = {request_per_minute:.2f}")
                task_results = run_task(task=task, fixed_prompt_text=fixed_prompt_text,
                                        cache=cache, converter=converter, cut_prompt_examples=cut_prompt_examples, task_idx=task_idx,
                                        engine=engine, max_tokens=max_tokens)
                results.append(task_results)
                is_success = True
                break
            except openai.error.InvalidRequestError as e:
                
                logging.info(
                    f"InvalidRequestError: {e}, trying with shorter prompt (cut_prompt_examples={cut_prompt_examples + 1 if cut_prompt_examples is not None else 1})")
                # sleep for a bit to further avoid rate limit exceeded exceptions
                time.sleep(5)
                continue
            except Exception as e:  # something else went wrong
                logging.info(f"Task {task_idx} failed: {e}")
                break

        if is_success and keep_writing_output:
            pd.DataFrame(results).to_json(
                output_file_path, orient='records', lines=True)

    print(
        f"Ran {len(results)} out of {len(tasks)} tasks ({len(results) / len(tasks):.2%})")
    pd.DataFrame(results).to_json(
        output_file_path, orient='records', lines=True)


def run_task(task: dict, fixed_prompt_text: str, cache: dict, converter: GraphPythonConverter, task_idx: int, engine: str, max_tokens: int, cut_prompt_examples: int = None) -> dict:
    """Runs the task, and returns the results.

    Args:
        task (dict): The task input
        fixed_prompt_text (str): Used for cases where the input prompt is fixed
        cache (dict): cache of previous results
        converter (GraphPythonConverter): A graph-python converter to parse results
        cut_prompt_examples (int, optional): If provided, the first `cut_prompt_examples` examples are 
                                             deleted. Prevents 4096 errors. Defaults to None.

    Returns:
        dict: A dictionary with the results.
    """
    start_time = time.time()
    prompt_text = fixed_prompt_text if fixed_prompt_text is not None else task['prompt']

    if cut_prompt_examples is not None:
        prompt_text_parts = prompt_text.split(END)
        prompt_text = END.join(prompt_text_parts[cut_prompt_examples:])

    if task['input_prompt_code'] in cache:
        logging.info(
            f"Task {task_idx} > Using cached result for {task['input_prompt_code']}")
        codex_response = cache[task['input_prompt_code']]["codex_response"]
    else:
        codex_response = query_codex(task, prompt_text, engine, max_tokens=max_tokens)


    completed_code = get_completed_code(task, codex_response)

    graph = converter.python_to_graph(completed_code)

    task_results = {k: v for (k, v) in task.items()}
    task_results["codex_response"] = codex_response
    task_results["generated_code"] = completed_code

    task_results["generated_graph"] = graph
    task_results["elapsed_time"] = time.time() - start_time

    return task_results


def maintain_request_per_minute(num_requests: int, time_begin: float, max_requests_per_min: int, task_idx: int) -> float:
    request_per_minute = get_request_per_minute(num_requests, time_begin)
    logging.info("\n")
    while request_per_minute > max_requests_per_min:
        logging.info(
            f"Task {task_idx} > Sleeping! (Requests/minute = {request_per_minute:.2f} > {max_requests_per_min:.2f})")
        time.sleep(1)
        request_per_minute = get_request_per_minute(
            num_requests, time_begin)
    return request_per_minute


def read_prompt(prompt_path):
    if prompt_path is None:
        return None
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt


def load_cache(output_file_path: str):
    """We don't want to query codex repeatedly for the same input. If an output file exists, this
    function creates a "cache" of the results.
    The cache is implemented as a hashmap keyed by `input_prompt_code`, and maps to the 
    entire output entry

    Args:
        output_file_path (str): _description_
    """
    if not os.path.exists(output_file_path):
        return {}
    else:
        # make a backup of the file already there
        shutil.copyfile(output_file_path, output_file_path + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.copy(output_file_path, output_file_path + ".bak")
        cache_data = pd.read_json(
            output_file_path, orient='records', lines=True)
        cache = {row['input_prompt_code']: row.to_dict()
                 for _, row in cache_data.iterrows()}
        return cache


def query_codex(task: dict, prompt_text: str, engine: str, max_tokens: int):
    prompt = f"{prompt_text} {task['input_prompt_code']}"
    response = OpenaiAPIWrapper.call(
        prompt=prompt, max_tokens=max_tokens, engine=engine)
    return response


def get_completed_code(task: dict, codex_response: dict) -> str:
    completed_code = OpenaiAPIWrapper.parse_response(codex_response)
    all_code = f"{task['input_prompt_code']}{completed_code}"
    # NOTE: space is already taken care of, no need to add it again, otherwise indentation will be off
    return all_code


def get_request_per_minute(num_request: int, begin_time: float) -> float:
    elapsed_time = time.time() - begin_time
    request_per_minute = (num_request / elapsed_time) * 60
    return request_per_minute


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file_path", type=str, required=True)
    parser.add_argument("--num_tasks", type=int, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str,
                        required=False, default=None)
    parser.add_argument("--job_type", type=str, required=True,
                        choices=ConverterFactory.supported_converters)
    parser.add_argument("--keep_writing_output",
                        action="store_true", default=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument("--max_requests_per_min", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=280)
    args = parser.parse_args()

    run(task_file_path=args.task_file_path, num_tasks=args.num_tasks,
        output_file_path=args.output_file_path, prompt_path=args.prompt_path,
         keep_writing_output=args.keep_writing_output, engine=args.engine,
            max_requests_per_min=args.max_requests_per_min, max_tokens=args.max_tokens)


