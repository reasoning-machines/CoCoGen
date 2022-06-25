# compuates the edge prediction precision, recall, and f1
from collections import defaultdict
from graphviz import Graph
import pandas as pd
from typing import Set

from src.eval.edge_pred_eval import get_edge_pred_eval_metrics
from src.utils.algo_utils import GraphAlgos

def run_edge_pred_spot_check(path: str):
    data = pd.read_json(path, lines=True, orient="records")
    gold_edges, predicted_edges = [], []

    for i, row in data.iterrows():
        try:
            if "relations" not in row["generated_graph"]:
                continue
            gold_edges, gold_edges_dict = get_edges(row["reference_graph"])
            gold_order =  GraphAlgos.topo_sort(GraphAlgos.to_nx_graph(gold_edges_dict))

            predicted_edges, pred_edges_dict = get_edges(row["generated_graph"])
            pred_order =  GraphAlgos.topo_sort(GraphAlgos.to_nx_graph(pred_edges_dict))
        
            precision, recall, f1, gold_edges, predicted_edges = get_edge_pred_eval_metrics(gold_edges, predicted_edges)

            print("-------------------------------------------------------")
            print(f"Scenario = {row['scenario']}")
            print("****")
            print(f"Precision = {precision}, Recall = {recall}, F1 = {f1}")
            print("****")
            pprint_edges_given_order("Gold edges", gold_edges_dict, gold_order)
            print("****")
            pprint_edges_given_order("Pred edges", pred_edges_dict, pred_order)
            gold_edges = set(gold_edges)
            predicted_edges = set(predicted_edges)
            common_edges = gold_edges & predicted_edges
            missed_edges = gold_edges - predicted_edges
            extra_edges = predicted_edges - gold_edges
            print("****")
            pprint_edges("Common edges", common_edges)
            print("****")
            pprint_edges("Missed edges", missed_edges)
            print("****")
            pprint_edges("Extra edges", extra_edges)
            print("-------------------------------------------------------")
            print("-------------------------------------------------------")
            input()
        except Exception as e:
            continue


def pprint_edges(header: str, edges: Set):
    print(f"{header} =>")
    for edge in edges:
        print(f"{edge[0]} -> {edge[1]}")

def pprint_edges_given_order(header: str, edges_dict: dict, order: list):
    print(f"{header} =>")
    for edge in order:
        for dest in edges_dict[edge]:
            print(f"{edge} -> {dest}")


def get_edges(graph: dict) -> Set:

    step_desc_dict = dict()
    for step in graph["schema"]:
        a, b = step.split(":")
        step_desc_dict[a.strip()] = b.strip()
    edges = []
    edge_dict = defaultdict(list)
    for edge in graph["relations"]:
        edge = edge.split(" -> ")
        edges.append((step_desc_dict[edge[0]], step_desc_dict[edge[1]]))
        edge_dict[step_desc_dict[edge[0]]].append(step_desc_dict[edge[1]])
    return edges, edge_dict

if __name__ == '__main__':
    import sys
    run_edge_pred_spot_check(sys.argv[1])