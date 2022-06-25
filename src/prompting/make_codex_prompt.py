import pandas as pd

from src.prompting.constants import END

def make_prompt(file_path: str, n_examples, seed: int = 0):
    data = pd.read_json(file_path, orient='records', lines=True)
    samples = data.sample(n_examples, random_state=seed)
    prompt = ""
    for i, sample in samples.iterrows():
        prompt += sample["reference_code"]
        prompt += f"{END}\n\n"
    
    print(prompt)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    else:
        seed = 0
    make_prompt(sys.argv[1], int(sys.argv[2]), seed)