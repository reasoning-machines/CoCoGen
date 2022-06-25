import os
import pickle
import logging
import pathlib
import random
from typing import List
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from src.eval.proscript_struct_eval import to_nx_graph
from src.eval.struct_eval import calculate_jsd
from src.eval.graph_metrics import PyDotMetrics, NxMetrics

logging.basicConfig(level=logging.INFO)

random.seed(42)

def make_pairs(inpath: str, outpath: str, n_samples: int = None, n_workers: int = 40):
    data = get_graphs_with_meta_info(inpath, outpath)

    if n_samples is None:
        n_samples = len(data)

    arguments_for_parallel_processing = []
    for i, row in tqdm(enumerate(data), total=len(data), desc="Making paired training data"):
        arguments_for_parallel_processing.append((row, i, data, n_samples, outpath))
    

    Parallel(n_jobs=n_workers)(delayed(make_data_for_single_graph)(*arg) for arg in arguments_for_parallel_processing)

def make_data_for_single_graph(graph: dict, graph_idx: int, all_graphs: List[dict], n_samples: int, outpath: str) -> List[dict]:
    if os.path.exists(f"{args.outpath}/graph_diff_pairs_{graph_idx}.pkl"):
        logging.info(f"{args.outpath}/graph_diff_pairs_{graph_idx}.pkl already exists, skipping")
        return
    # Creates example pairs for a single graph
    pair_idxs = [random.randint(0, len(all_graphs)-1) for _ in range(n_samples)]
    pair_idxs = [idx for idx in pair_idxs if idx != graph_idx]
    pairs = []
    for pair_idx in tqdm(pair_idxs, total=len(pair_idxs), desc=f"Making paired training data for {graph_idx}"):
        sample_to_diff_with = all_graphs[pair_idx]
        diff_info = get_graph_diff(graph, sample_to_diff_with)
        tmp = {
            "idx_1": graph_idx,
            "idx_2": pair_idx,
            "diff_info": diff_info,
            "graph1": graph,
            "graph2": sample_to_diff_with
        }
        pairs.append(tmp)
    with open(f"{args.outpath}/graph_diff_pairs_{graph_idx}.pkl", "wb") as f:
        pickle.dump(pairs, f)
    return pairs


def get_graphs_with_meta_info(
    inpath: str, outpath: str
):
    inpath_filename = pathlib.Path(inpath).name
    outpath_dot = pathlib.Path(outpath) / inpath_filename.replace(".jsonl", "_with_graphs_meta.pkl")
    if outpath_dot.exists():
        logging.info(f"Loading graphs from {outpath_dot}")
        with open(outpath_dot, "rb") as f:
            data_with_graphs = pickle.load(f)
            return data_with_graphs

    data_with_graphs = read_graphs(inpath, outpath)

    for i, row in enumerate(data_with_graphs):
        degree_stats = NxMetrics.calc_degree_summary_stats([row["nx_graph"]])
        row["avg_degree"] = degree_stats[0]
        num_nodes, num_edges = PyDotMetrics.calc_num_nodes_num_edges([
                                                                     row["dot_graph"]])
        num_nodes = int(num_nodes)  # the function returns an average over multiple graphs, but we only have one
        num_edges = int(num_edges)
        row["num_nodes"] = num_nodes
        row["num_edges"] = num_edges

        indegree_dist, outdegree_dist, degree_dist = NxMetrics.get_in_out_degree_distribution([
                                                                                              row["nx_graph"]])

        row["indegree_dist"] = indegree_dist
        row["outdegree_dist"] = outdegree_dist
        row["degree_dist"] = degree_dist
        row["indegree_gte_2"] = sum([indegree_dist[i] for i in range(2, num_nodes + 1)])
        row["outdegree_gte_2"] = sum([outdegree_dist[i] for i in range(2, num_nodes + 1)])
        row["degree_gt_3"] = sum([degree_dist[i] for i in range(3, num_nodes + 1)])

        row["indegree_gte_2_ratio"] = row["indegree_gte_2"] / num_nodes
        row["outdegree_gte_2_ratio"] = row["outdegree_gte_2"] / num_nodes
        row["degree_gt_3_ratio"] = row["degree_gt_3"] / num_nodes

    
    
    inpath_filename = pathlib.Path(inpath).name
    outpath_dot = pathlib.Path(outpath) / inpath_filename.replace(".jsonl", "_with_graphs_meta.pkl")
    pickle.dump(data_with_graphs, open(outpath_dot, "wb"))
    return data_with_graphs


def get_graph_diff(graph1_info: dict, graph2_info: dict) -> dict:
    # find edist
    edist, _ = NxMetrics.graph_edist([graph1_info["nx_graph"]], [graph2_info["nx_graph"]], timeout=2, verbose=False)
    indegree_jsd = calculate_jsd(graph1_info["indegree_dist"], graph2_info["indegree_dist"], verbose=False)
    outdegree_jsd = calculate_jsd(graph1_info["outdegree_dist"], graph2_info["outdegree_dist"], verbose=False)
    degree_jsd = calculate_jsd(graph1_info["degree_dist"], graph2_info["degree_dist"], verbose=False)
    isomorphism = NxMetrics.isomorphism([graph1_info["nx_graph"]], [graph2_info["nx_graph"]], verbose=False)

    # diff in num nodes etc.
    num_nodes_diff = abs(graph1_info["num_nodes"] - graph2_info["num_nodes"])
    num_edges_diff = abs(graph1_info["num_edges"] - graph2_info["num_edges"])
    avg_degree_diff = abs(graph1_info["avg_degree"] - graph2_info["avg_degree"])
    indegree_gte_2_diff = abs(graph1_info["indegree_gte_2"] - graph2_info["indegree_gte_2"])
    outdegree_gte_2_diff = abs(graph1_info["outdegree_gte_2"] - graph2_info["outdegree_gte_2"])
    degree_gt_3_diff = abs(graph1_info["degree_gt_3"] - graph2_info["degree_gt_3"])
    indegree_gte_2_ratio_diff = abs(graph1_info["indegree_gte_2_ratio"] - graph2_info["indegree_gte_2_ratio"])
    outdegree_gte_2_ratio_diff = abs(graph1_info["outdegree_gte_2_ratio"] - graph2_info["outdegree_gte_2_ratio"])
    degree_gt_3_ratio_diff = abs(graph1_info["degree_gt_3_ratio"] - graph2_info["degree_gt_3_ratio"])

    diff_info = {
        "edist": edist,
        "indegree_jsd": indegree_jsd,
        "outdegree_jsd": outdegree_jsd,
        "degree_jsd": degree_jsd,
        "num_nodes_diff": num_nodes_diff,
        "num_edges_diff": num_edges_diff,
        "avg_degree_diff": avg_degree_diff,
        "indegree_gte_2_diff": indegree_gte_2_diff,
        "outdegree_gte_2_diff": outdegree_gte_2_diff,
        "degree_gt_3_diff": degree_gt_3_diff,
        "indegree_gte_2_ratio_diff": indegree_gte_2_ratio_diff,
        "outdegree_gte_2_ratio_diff": outdegree_gte_2_ratio_diff,
        "degree_gt_3_ratio_diff": degree_gt_3_ratio_diff,
    }

    return diff_info


def read_graphs(inpath: str, outpath: str):

    inpath_filename = pathlib.Path(inpath).name
    outpath_graph = pathlib.Path(outpath)/ inpath_filename.replace(".jsonl", "_nx_graph.pkl")
    outpath_dot = pathlib.Path(outpath)/ \
        inpath_filename.replace(".jsonl", "_dot_graph.pkl")

    data = pd.read_json(inpath, orient="records", lines=True)
    if outpath_graph.exists() and outpath_dot.exists():
        logging.info(f"Loading graphs from {outpath_graph} and {outpath_dot}")
        with open(outpath_graph, "rb") as f:
            nx_graphs = pickle.load(f)
        with open(outpath_dot, "rb") as f:
            dot_graphs = pickle.load(f)

    else:
        nx_graphs = []
        dot_graphs = []
        logging.info(
            f"Graphs pickle files not found at {outpath_graph} and {outpath_dot}, generating them from scratch")

        for _, row in tqdm(data.iterrows(), total=len(data), desc="Parsing graphs"):
            pred_nx_graph, pred_dot_graph = to_nx_graph(
                row["flatten_output_for_script_generation"], return_dot=True)
            if pred_nx_graph is not None and pred_dot_graph is not None:
                nx_graphs.append(pred_nx_graph)
                dot_graphs.append(pred_dot_graph)
        pickle.dump(nx_graphs, open(outpath_graph, "wb"))
        pickle.dump(dot_graphs, open(outpath_dot, "wb"))

    # attach these graphs to the dataframe

    data_with_graphs = []
    for i, row in tqdm(data.iterrows(), total=len(data), desc="Attaching graphs"):
        tmp = row.to_dict()
        tmp["nx_graph"] = nx_graphs[i]
        tmp["dot_graph"] = dot_graphs[i]
        data_with_graphs.append(tmp)

    return data_with_graphs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()
    
    paired_diffs = make_pairs(inpath=args.inpath, outpath=args.outpath, n_samples=args.n_samples)

    # with open(args.outpath + "/graph_diff_pairs.pkl", "wb") as f:
    #     pickle.dump(paired_diffs, f)
        
