from collections import defaultdict
import json
from typing import Dict, List, Union
import pandas as pd

from src.eval.content_eval import NodeMetricsCalculator
from src.utils.algo_utils import GraphAlgos

def extract_nodes_as_str(graph_str: str) -> str:
    def _get_nodes_list(schema: Union[str, Dict]) -> List[str]:
        """Given a schema of the form
        [
                "step0: decided to buy ingredients for cookies",
                "step1: go to the store",
                "step2: go to the cookie isle",
                "step3: grab the chocolate chips",
                "step4: grab the flour",
                "step5: grab the sugar",
                "step6: grab the butter",
                "step7: grab the eggs",
                "step8: buy ingredients for cookies"
        ]
        OR a string of the form
        "step0: look for polls information; step1: find polls information; step2: skim through the information; step3: find the results page's location; step4: go to the results page; step5: start reading the results; step6: glance through results; step0 -> step1; step1 -> step2; step2 -> step3; step3 -> step4; step4 -> step5; step5 -> step6"
        Extract the nodes as a list of string:
            [look for polls information, find polls information, ..., glance through results]
        """
        if isinstance(schema, str):
            steps = schema.split("; ")
            relations = None
        elif isinstance(schema, dict):
            steps = schema["schema"]
            relations = schema["relations"]
        else:
            raise ValueError(f"schema must be either a string or a dictionary, got {type(schema)} ({schema})")

        # First create the steps
        step_id_to_step = dict()
        for line in steps:
            line = line.strip()
            if ":" in line:
                step_id, step = line.split(":")
                step_id_to_step[step_id.strip()] = step.strip()
        

        # Now, create the edges either from relations array or by parsing
        step_id_to_next_steps = defaultdict(list)

        if relations is not None:  # need to create step_id_to_next_steps 
            for relation in relations:
                from_step, to_step = relation.split("->")
                step_id_to_next_steps[from_step.strip()].append(to_step.strip())
        else:
            for line in steps:  # create it from string representation
                line = line.strip()
                if "->" in line:
                    from_step, to_step = line.split("->")
                    step_id_to_next_steps[from_step.strip()].append(to_step.strip())
    
        step_graph = GraphAlgos.to_nx_graph(step_id_to_next_steps)
        topo_order = GraphAlgos.topo_sort(step_graph)
        nodes = [step_id_to_step[step_id] for step_id in topo_order]

        return nodes

    try:
        nodes = _get_nodes_list(graph_str)
    except Exception as e:
        return "NONE"

    # remove periods
    cleaned_nodes = []
    for node in nodes:
        if len(node) > 0 and node[-1] == ".":
            node = node[:-1]
        cleaned_nodes.append(node.lower().strip())
        

    return ". ".join(cleaned_nodes) + "."


def find_ref_col(columns):
    possible_ref_columns = ["answer", "target", "reference_graph"]
    for col in possible_ref_columns:
        if col in columns:
            return col

def find_pred_col(columns):
    possible_pred_columns = ["greedy_generated_answer_from_question", "pred", "generated_graph"]
    for col in possible_pred_columns:
        if col in columns:
            return col


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate metrics for a given dataset")
    parser.add_argument("--outputs", type=str, help="Path to the dataset")
    parser.add_argument("--ref_col", type=str, help="Column name of the reference", default=None)
    parser.add_argument("--pred_col", type=str, help="Column name of the prediction", default=None)
    parser.add_argument("--report", type=str)

    args = parser.parse_args()
    data = pd.read_json(args.outputs, orient='records', lines=True)

    if args.ref_col is None:
        args.ref_col = find_ref_col(data.columns)
    
    if args.pred_col is None:
        args.pred_col = find_pred_col(data.columns)

    metrics_calculator = NodeMetricsCalculator()

    metrics = metrics_calculator.run(data=data, ref_col=args.ref_col, pred_col=args.pred_col, get_nodes_func=extract_nodes_as_str)
    with open(args.report, "w") as f:
        json.dump(metrics, f, indent=2)
