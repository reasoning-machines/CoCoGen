# compuates the edge prediction precision, recall, and f1
import json
from pprint import pprint
import pandas as pd
from typing import List, Tuple


def run_edge_pred(path: str, report_path: str):
    data = pd.read_json(path, lines=True, orient="records")
    gold_edges, predicted_edges = [], []

    for i, row in data.iterrows():
        gold_edges = row["reference_graph"]["relations"]
        predicted_edges = row["generated_graph"]
        if predicted_edges and "relations" in predicted_edges:
            predicted_edges = predicted_edges["relations"]
            precision, recall, f1, _, _  = get_edge_pred_eval_metrics(gold_edges, predicted_edges)
        else:
            precision, recall, f1 = 0, 0, 0
        
        data.loc[i, "edge_pred_precision"] = precision
        data.loc[i, "edge_pred_recall"] = recall
        data.loc[i, "edge_pred_f1"] = f1

    report = {
        "edge_pred_precision": data["edge_pred_precision"].mean(),
        "edge_pred_recall": data["edge_pred_recall"].mean(),
        "edge_pred_f1": data["edge_pred_f1"].mean(),
        "#samples": len(data) // 100,
    }
    
    report = {k: round(v * 100, 2) for k, v in report.items()}
    pprint(report)
    if report_path:
        with open(report_path, "w") as f:
            f.write(json.dumps(report) + "\n")


def get_edge_pred_eval_metrics(
    gold_edges: List[str],
    predicted_edges: List[str],
) -> Tuple[float, float, float]:
    """
    Computes the precision, recall, and f1 of the predicted edges against the gold edges.

    Args:
        gold_edges: The gold edges.
        predicted_edges: The predicted edges.

    Returns:
        The precision, recall, and f1 of the predicted edges against the gold edges.
    """
    # print(f"Gold edges = {gold_edges}")
    # print(f"Predicted edges = {predicted_edges}")
    gold_edges = set(gold_edges)
    predicted_edges = set(predicted_edges)

    precision = len(gold_edges & predicted_edges) / len(predicted_edges) if len(predicted_edges) > 0 else 0
    recall = len(gold_edges & predicted_edges) / len(gold_edges) if len(gold_edges) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0


    # print(f"Precision = {precision}, Recall = {recall}, F1 = {f1}")

    # print("-------------------------------------------------------")
    # print("-------------------------------------------------------")
    return precision, recall, f1, gold_edges, predicted_edges


if __name__ == '__main__':
    import sys
    run_edge_pred(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)