# Taken almost as-is from https://github.com/VHellendoorn/Code-LMs/blob/main/Evaluation/eval_codex_all.py

# calculates the perplexity of a given piece of text from codex

import argparse
import glob
import json
import os
import time
import math
import openai
from tqdm import tqdm

# The private OpenAI API key needs to be an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
# As instructed here: https://community.openai.com/t/token-logprobs-when-echo-is-true/9626/2
# "Transformer models don’t predict the probability of the first token. If you want to get the probability 
# for your first token you can try to use <|endoftext|> as the first token as a workaround."
endoftext_token = '<|endoftext|>'

def ppl(avg_logprob):
    return 2 ** (-avg_logprob / math.log(2))

def call_codex(txt: str, engine: str, save_probs: bool):
    eos_code_str = endoftext_token + txt
    # engine: 'davinci-codex' is currently the best codex model
    # max_tokens=0 means that we don't want the model to generate additional tokens
    # logprobs=0 means that we don't want the logprobs of the alternative tokens, only the actual tokens
    # echo=True means that we want the model to echo our prompt, in addition to our (not existing) completion
    completion = openai.Completion.create(engine=engine, prompt=eos_code_str,
                                          max_tokens=0,
                                          temperature=0.0,
                                          logprobs=0,
                                          n=1,
                                          echo=True)
    
    c = completion.choices[0]
    # skipping the <|endoftext|> token
    sum_logprobs = sum(c.logprobs.token_logprobs[1:])
    num_tokens = len(c.logprobs.token_logprobs[1:])
    if save_probs:
        saved_probs = {
            'text': txt,
            'tokens': c.logprobs.tokens[1:],
            'logprobs': c.logprobs.token_logprobs[1:],
            'sum_logprobs': sum_logprobs
        }
    else:
        saved_probs = None    

    return sum_logprobs, num_tokens, saved_probs

def get_ppl_for_dir(dir: str, engine: str, save_probs: bool, requests_per_min: int) -> dict:
    # read all data from the directory
    files = glob.glob(os.path.join(dir, '**/*'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
        
    log_probs_sum = 0
    tokens_count = 0
    ignored_files = []
    print(files)
    num_requests = 0
    request_per_minute = None
    time_begin = time.time()

    for file in tqdm(files, total=len(files)):
        print(f'Processing {file}, requests_per_minute={request_per_minute}')
        try:
            with open(file, 'r') as f:
                code_str = f.read()
            
            num_requests += 1
            request_per_minute = get_request_per_minute(num_requests, time_begin)
            
            print(f"requests per minute: {request_per_minute:.2f}")
            while request_per_minute > requests_per_min:
                time.sleep(1)
                request_per_minute = get_request_per_minute(num_requests, time_begin)
                

            logprobs_sum, logprobs_count, per_token_probs = call_codex(code_str, save_probs=save_probs, engine=engine)

            log_probs_sum += logprobs_sum
            tokens_count += logprobs_count

        except Exception as e:
            print(f'EXCEPTION in file {file}: {e}')
            print(e)
            ignored_files.append(file)
            # OpenAI limits the request rate to 20/min
            continue


    print(f'\n\n\nlogprobs sum: {log_probs_sum}')
    print(f'total tokens: {tokens_count}')
    print(f'Average loss: {-log_probs_sum / tokens_count}')
    print(f'Perplexity: {ppl(log_probs_sum / tokens_count)}')
    print(f'Ignored files:')
    
    print()
    print('\033[1;34m--------------------------------------------------------------------------------\033[0m')
    print('\033[1;34m--------------------------------------------------------------------------------\033[0m')
    print()
    
    return {
        'log_probs_sum': log_probs_sum,
        'tokens_count': tokens_count,
        'average_loss': -log_probs_sum / tokens_count,
        'perplexity': ppl(log_probs_sum / tokens_count),
    }
    
def get_request_per_minute(num_request: int, begin_time: float) -> float:
    elapsed_time = time.time() - begin_time
    request_per_minute = (num_request / elapsed_time) * 60
    return request_per_minute

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_config', type=str, help='Path to a json file that contains folder to be evaluated with the corresponding model', required=False)
    parser.add_argument('--save_probs', action='store_true', help='Save per-token probabilities in the output file', default=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--requests_per_min', type=int, required=False, default=10)
    args = parser.parse_args()
    with open(args.job_config, 'r') as f:
        job_config = json.load(f)
    
    results = dict()
    for task, task_details in job_config.items():
        print(f'Evaluating {task} - {task_details["path"]} with {task_details["engine"]}')
        scores = get_ppl_for_dir(dir=task_details['path'], engine=task_details['engine'], save_probs=args.save_probs, requests_per_min=args.requests_per_min)
        results[task] = scores
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)