import argparse
import json
import os.path

import pandas as pd
from src.prompting.constants import END, END_LINE


def make_prompt(file_path: str, n_examples, prompt_id, sep=END):
    with open(os.path.dirname(file_path) + '/prompt_mapping.json', 'r') as f:
        prompt_mapping = json.load(f)

    data = pd.read_json(file_path, orient='records', lines=True)
    if prompt_id in prompt_mapping:
        # print(f'prompt ids: {prompt_mapping[prompt_id]}')
        samples = {}
        # make sure the order is the same
        for i in range(len(data)):
            sample = data.loc[i]
            if int(sample['id']) in prompt_mapping[prompt_id]:
                samples[sample['id']] = sample
        samples = [samples[i] for i in prompt_mapping[prompt_id]]
    else:
        print('randomly selected')
        samples = data.sample(n_examples)
        samples = [samples.loc[i] for i in range(len(samples))]

    prompt = ""
    for sample in samples:
        prompt += sample["reference_code"]
        prompt += f"{sep}\n\n"

    print(prompt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Path to the file to be used as prompt")
    parser.add_argument("--n_examples", type=int, help="Number of examples to be used as prompt")
    parser.add_argument("--prompt_id", type=str, help="ID of the prompt")
    parser.add_argument("--sep", type=str, help='sample separator', default='end')
    args = parser.parse_args()
    make_prompt(args.file_path, args.n_examples, args.prompt_id, sep=END if args.sep == 'end' else END_LINE)