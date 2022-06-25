import json
from typing import Dict, List, Tuple
from numpy import round_
import pydot
import networkx as nx
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon

from src.eval.graph_metrics import PyDotMetrics, NxMetrics


def evaluate_graphs(output_file_path: str, gold_col: str, pred_col: str, to_nx_graph, gedit_timeout: int = 3, verbose: bool = False) -> dict:
    """Evaluates the graphs in the given columns.

    Args:
        df (pd.DataFrame): Contains all the data
        pred_col (str): Column name of the predicted graph
        ref_col (str): Column name of the reference graph
        to_nx_graph (function): Function to convert the graph representation to networkx graph
        gedit_timeout (int, optional): GED calculation can be slow. The library accepts a timeout and after these many seconds the current GED value is returned.
        verbose (bool, optional): Prints out the metrics on stdout.

    Returns:
        dict: A report dictionary.
    """
    data = pd.read_json(output_file_path, orient='records', lines=True)

    pred_nx_graphs, pred_pydot_graphs, target_nx_graphs, target_pydot_graphs = parse_dot_column_to_graph(
        data, pred_col=pred_col, ref_col=gold_col, to_nx_graph=to_nx_graph)

    pred_indegree_dist, pred_outdegree_dist, pred_degree_dist = NxMetrics.get_in_out_degree_distribution(
        pred_nx_graphs)
    target_indegree_dist, target_outdegree_dist, target_degree_dist = NxMetrics.get_in_out_degree_distribution(
        target_nx_graphs)

    avg_degree_pred, median_degree_pred, avg_degree_overall_pred, median_degree_overall_pred = NxMetrics.calc_degree_summary_stats(
        pred_nx_graphs)
    avg_degree_target, median_degree_target, avg_degree_overall_target, median_degree_overall_target = NxMetrics.calc_degree_summary_stats(
        target_nx_graphs)

    avg_pred_num_nodes, avg_pred_num_edges = PyDotMetrics.calc_num_nodes_num_edges(
        pred_pydot_graphs)
    avg_target_num_nodes, avg_target_num_edges = PyDotMetrics.calc_num_nodes_num_edges(
        target_pydot_graphs)

    isomorphism = NxMetrics.isomorphism(pred_nx_graphs, target_nx_graphs)

    edist = NxMetrics.graph_edist(
        pred_nx_graphs, target_nx_graphs, timeout=gedit_timeout)
    # show the metrics side by side with equal width and format to 2 decimal places
    print("\n\n\n")

    def round_vals(x): return round_(x, 3)
    if verbose:
        print(
            f"Average degree        gold = {round_vals(avg_degree_target):.2f} | pred = {round_vals(avg_degree_pred):.2f}")
        print(
            f"Median degree         gold = {round_vals(median_degree_target):.2f} | pred = {round_vals(median_degree_pred):.2f}")
        print(
            f"Avg. Degree Overall   gold = {round_vals(avg_degree_overall_target):.2f} | pred = {round_vals(avg_degree_overall_pred):.2f}")
        print(
            f"Median Degree Overall gold = {round_vals(median_degree_overall_target):.2f} | pred = {round_vals(median_degree_overall_pred):.2f}")
        print(
            f"Average #nodes        gold = {round_vals(avg_target_num_nodes):.2f} | pred = {round_vals(avg_pred_num_nodes):.2f}")
        print(
            f"Average #edges        gold = {round_vals(avg_target_num_edges):.2f} | pred = {round_vals(avg_pred_num_edges):.2f}")

        print("\n\n\n")
        print(f"Isomorphism = {isomorphism}")
        print(
            f"Graph edit Distance (GED) = {edist[0]} (#graph-pairs = {edist[1]}, timeout = {gedit_timeout})")

    report = {
        "degree_stats": {
            "gold": {
                "avg_degree": round_vals(avg_degree_target),
                "median_degree": median_degree_target,
                "avg_degree_overall": round_vals(avg_degree_overall_target),
                "median_degree_overall": median_degree_overall_target
            },
            "pred": {
                "avg_degree": round_vals(avg_degree_pred),
                "median_degree": median_degree_pred,
                "avg_degree_overall": round_vals(avg_degree_overall_pred),
                "median_degree_overall": median_degree_overall_pred
            }
        },
        "num_nodes_edges": {
            "gold": {
                "avg_num_nodes": round_vals(avg_target_num_nodes),
                "avg_num_edges": round_vals(avg_target_num_edges)
            },
            "pred": {
                "avg_num_nodes": round_vals(avg_pred_num_nodes),
                "avg_num_edges": round_vals(avg_pred_num_edges)
            }
        },
        "isomorphism": isomorphism,
        "ged": {
            "graph_edit_distance": edist[0],
            "num_graph_pairs": edist[1],
            "timeout": gedit_timeout
        },
        "degree_dist": {
            "gold": {
                "indegree_dist": target_indegree_dist,
                "outdegree_dist": target_outdegree_dist,
                "degree_dist": target_degree_dist
            },
            "pred": {
                "indegree_dist": pred_indegree_dist,
                "outdegree_dist": pred_outdegree_dist,
                "degree_dist": pred_degree_dist
            }
        },
        "jsd": {
            "indegree_jsd": calculate_jsd(target_indegree_dist, pred_indegree_dist),
            "outdegree_jsd": calculate_jsd(target_outdegree_dist, pred_outdegree_dist),
            "degree_jsd": calculate_jsd(target_degree_dist, pred_degree_dist)
        },
        "parse_rate": round_vals(len(pred_nx_graphs) * 100 / len(data))
    }
    return report

def parse_dot_column_to_graph(df: pd.DataFrame, pred_col: str, ref_col: str, to_nx_graph) -> Tuple[nx.Graph, pydot.Graph, nx.Graph, pydot.Graph]:
    """ Parses proscript graph representation to networkx and dot representations.

    Args:
        df (pd.DataFrame): Contains all the data
        pred_col (str): Column name of the predicted graph
        ref_col (str): Column name of the reference graph
        to_nx_graph (function): Function to convert the graph representation to networkx graph

    Returns:
        Tuple[nx.Graph, pydot.Graph, nx.Graph, pydot.Graph]: (generated graph nx, generated graph pydot, reference graph nx, reference graph pydot)
    """
    num_parseable = 0
    generated_nx_graphs = []
    generated_pydot_graphs = []
    reference_nx_graphs = []
    reference_pydot_graphs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing graphs"):
        try:
            pred_nx_graph, pred_dot_graph = to_nx_graph(
                row[pred_col], return_dot=True)
            ref_nx_graph, ref_dot_graph = to_nx_graph(
                row[ref_col], return_dot=True)

            # check that all are non None
            if pred_nx_graph is not None and pred_dot_graph is not None and ref_nx_graph is not None and ref_dot_graph is not None:
                generated_nx_graphs.append(pred_nx_graph)
                generated_pydot_graphs.append(pred_dot_graph)
                reference_nx_graphs.append(ref_nx_graph)
                reference_pydot_graphs.append(ref_dot_graph)
                num_parseable += 1
        except Exception as e:
            print(e)
            continue

    frac_parseable = num_parseable / len(df)
    print(f"{num_parseable} graphs parsed out of {len(df)} ({frac_parseable:.2%}) from {pred_col}")
    return generated_nx_graphs, generated_pydot_graphs, reference_nx_graphs, reference_pydot_graphs


def patch_output(inpath: str, outpath: str):
    report = json.load(open(inpath, "r"))
    target_indegree_dist = report["degree_dist"]["gold"]["indegree_dist"]
    target_outdegree_dist = report["degree_dist"]["gold"]["outdegree_dist"]
    target_degree_dist = report["degree_dist"]["gold"]["degree_dist"]
    pred_indegree_dist = report["degree_dist"]["pred"]["indegree_dist"]
    pred_outdegree_dist = report["degree_dist"]["pred"]["outdegree_dist"]
    pred_degree_dist = report["degree_dist"]["pred"]["degree_dist"]
    print(f"Target indegree dist: {target_indegree_dist}")
    print(f"Target outdegree dist: {target_outdegree_dist}")
    report["jsd"] = {
            "indegree_jsd": calculate_jsd(target_indegree_dist, pred_indegree_dist),
            "outdegree_jsd": calculate_jsd(target_outdegree_dist, pred_outdegree_dist),
            "degree_jsd": calculate_jsd(target_degree_dist, pred_degree_dist)
        }
    json.dump(report, open(outpath, "w"), indent=4)

def calculate_jsd(dist1: Dict, dist2: Dict, verbose: bool = True) -> float:
    def _normalize(dist: Dict) -> Dict:
        return {k: v / sum(dist.values()) for k, v in dist.items()}

    def _add_support_if_not_present(dist: Dict, common_support: List[int]) -> List[float]:
        res = []
        for i in common_support:
            if i not in dist:
                res.append(0.0)
            else:
                res.append(dist[i])
        return res

    common_support = sorted(list(set(dist1.keys()) | set(dist2.keys())))
    dist1 = _add_support_if_not_present(_normalize(dist1), common_support)
    dist2 = _add_support_if_not_present(_normalize(dist2), common_support)
    if verbose:
        print(f"Common support: {common_support}")
        print(f"Dist1: {dist1}")
        print(f"Dist2: {dist2}")
        print(f"jsd = {jensenshannon(dist1, dist2)}")
        print("--")
    return jensenshannon(dist1, dist2)
    


if __name__ == "__main__":
    import sys
    patch_output(sys.argv[1], sys.argv[2])