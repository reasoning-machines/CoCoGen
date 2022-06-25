import json
from typing import Tuple, Union
import pydot
import networkx as nx

from src.eval.struct_eval import evaluate_graphs


def to_nx_graph(graph_repr: Union[str, dict], return_dot: bool) -> Tuple[nx.Graph, pydot.Graph]:
    """ Converts a proscript graph representation to a networkx graph."""
    if isinstance(graph_repr, str):
        return to_nx_graph_str(graph_repr, return_dot)
    elif isinstance(graph_repr, dict):
        return to_nx_graph_dict(graph_repr, return_dot)
    else:
        raise ValueError("graph_repr must be str or dict")


def to_nx_graph_str(graph_dot_rep: str, return_dot: bool) -> nx.Graph:
    parts = graph_dot_rep.split(";")
    nodes = [part.strip() for part in parts if ":" in part]
    nodes = [node.replace(": ", " [label=\"") + "\"]" for node in nodes]
    edges = [part.strip() for part in parts if "->" in part]
    dot_repr = "digraph G {\n" + \
        ";\n".join(nodes) + ";\n" + ";\n".join(edges) + "\n}"
    pydot_g = pydot.graph_from_dot_data(dot_repr)[0]
    g = nx.nx_pydot.from_pydot(pydot_g)
    if return_dot:
        return g, pydot_g
    return g


def to_nx_graph_dict(graph_repr: str, return_dot: bool) -> nx.Graph:
    """ Converts a proscript graph representation to a networkx graph."""
    nodes = graph_repr["schema"]
    nodes = [node.replace(": ", " [label=\"") + "\"]" for node in nodes]
    edges = graph_repr["relations"]
    dot_repr = "digraph G {\n" + \
        ";\n".join(nodes) + ";\n" + ";\n".join(edges) + "\n}"
    pydot_g = pydot.graph_from_dot_data(dot_repr)[0]
    g = nx.nx_pydot.from_pydot(pydot_g)
    if return_dot:
        return g, pydot_g
    return g


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=str)
    parser.add_argument("--gold_col", type=str, default="reference_graph")
    parser.add_argument("--pred_col", type=str, default="generated_graph")
    parser.add_argument("--gedit_timeout", type=int, default=3)
    parser.add_argument("--report", type=str)

    args = parser.parse_args()
    report = evaluate_graphs(output_file_path=args.outputs,
                             gold_col=args.gold_col, pred_col=args.pred_col, gedit_timeout=args.gedit_timeout,
                             to_nx_graph=to_nx_graph)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
