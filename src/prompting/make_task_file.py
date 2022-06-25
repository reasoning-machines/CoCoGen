import pandas as pd
from tqdm import tqdm
import json

from src.converters.get_converter import ConverterFactory


def make_task_file(args):

    data = read_data(args.inpath)
    converter = ConverterFactory.get_converter(args.job_type)

    res = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        try:
            py_source = converter.graph_to_python(row, prompt_part_only=False)
            if py_source is None:
                continue
            tmp = {k: v for (k, v) in row.items()}
            tmp["reference_code"] = py_source
            tmp["input_prompt_code"] = converter.graph_to_python(row, prompt_part_only=True)
            tmp["reference_graph"] = converter.python_to_graph(py_source)
        except Exception as e:
            raise e
            continue
        res.append(tmp)
    
    # successfully converted
    conversion_rate = len(res) / len(data)
    print(f"Converted {len(res)} out of {len(data)} rows ({conversion_rate:.2%})")
    pd.DataFrame(res).to_json(args.outpath, orient='records', lines=True)


def read_data(inpath):
    if "tsv" in inpath:
        data = pd.read_csv(inpath, sep="\t")
    elif "jsonl" in inpath:
        data = pd.read_json(inpath, orient='records', lines=True)
    elif "json" in inpath:
        with open(inpath, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                rows = []
                for record_id in data:
                    tmp = dict()
                    tmp["id"] = record_id
                    tmp.update(data[record_id])
                    rows.append(tmp)
                data = pd.DataFrame(rows)

    else:
        raise ValueError(f"Unknown input format: {inpath}")
    return data



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    make_task_file(args)
