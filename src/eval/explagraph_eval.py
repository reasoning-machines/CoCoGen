import pathlib
import pandas as pd
import tempfile

import subprocess


def run(output_path: str, tag: str):
    data = pd.read_json(output_path, lines=True, orient="records")
    predictions = []
    references = []
    for i, row in data.iterrows():
        predictions.append({
            "stance": row["generated_graph"]["stance"],
            "graph": row["generated_graph"]["graph"],
        })
        references.append({
            "stance": row["reference_graph"]["stance"],
            "graph": row["reference_graph"]["graph"],
            "belief": row["reference_graph"]["belief"],
            "argument": row["reference_graph"]["argument"],
        })
    predictions = pd.DataFrame(predictions)
    references = pd.DataFrame(references)
    references = references[["belief", "argument", "stance", "graph"]]
    predictions = predictions[["stance", "graph"]]
    # write to a random temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f_pred:
        predictions.to_csv(f_pred.name, index=False, header=False, sep="\t")
        print(predictions.head())
        print(f"Predictions temp file: {f_pred.name}")
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f_ref:
        references.to_csv(f_ref.name, index=False, header=False, sep="\t")
        print(references.head())
        print(f"References temp file: {f_ref.name}")
    
    output_path_dir = pathlib.Path(output_path).parent
    
    output_path_dir = pathlib.Path(*output_path_dir.parts[2:])

    parts = ["bash", "data/ExplaGraphs/eval_scripts/run_all.sh", f_pred.name, f_ref.name, output_path_dir]
    print(f"Running: {' '.join(map(str, parts))}")
    subprocess.Popen(parts).wait()
    

    # renmae the output fiilie to include the tag
    new_path = pathlib.Path(output_path).parent / f"{tag}.txt"
    report_path = pathlib.Path(output_path).parent / f"report.txt"
    pprint_results(report_path)
    print(f"Renaming {output_path} to {new_path}")
    report_path.rename(new_path)
    


    

def pprint_results(path):
    idx = {
        "struct_accuracy": 0,
        "eval_SeCA": 1,
        "g_bert_f1": 2,
        "ged": 3,
        "Edge Importance Accuracy (EA)": 4
    }
    scores = [0 for _ in range(len(idx))]
    with open(path) as f:
        for line in f:
            metric, value = line.strip().split("=")
            metric = metric.strip()
            if metric in idx:
                if idx != "ged":
                    scores[idx[metric]] = round(float(value) * 100, 2)
                else:
                    scores[idx[metric]] = round(float(value), 2)

    print(", ".join(map(str, scores)))

if __name__ == '__main__':
    import sys
    run(sys.argv[1], sys.argv[2])