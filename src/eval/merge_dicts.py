import json
import os
from pprint import pprint
from typing import Set
from glob import glob
import pandas as pd

test_input = {
  "degree_stats": {
    "gold": {
      "avg_degree": 1.86,
      "median_degree": 1.75,
      "avg_degree_overall": 1.863,
      "median_degree_overall": 2.0
    },
    "pred": {
      "avg_degree": 1.786,
      "median_degree": 1.7142857142857142,
      "avg_degree_overall": 1.787,
      "median_degree_overall": 2.0
    }
  },
  "num_nodes_edges": {
    "gold": {
      "avg_num_nodes": 7.446,
      "avg_num_edges": 6.936
    },
    "pred": {
      "avg_num_nodes": 7.446,
      "avg_num_edges": 6.653
    }
  },
  "isomorphism": 0.3510691198408752,
  "ged": {
    "graph_edit_distance": 3.003,
    "num_graph_pairs": 2011.0,
    "timeout": 3
  },
  "degree_dist": {
    "gold": {
      "indegree_dist": {
        "0": 2011,
        "1": 12131,
        "2": 727,
        "3": 70,
        "5": 11,
        "4": 23,
        "7": 1
      },
      "outdegree_dist": {
        "1": 12136,
        "2": 716,
        "0": 2011,
        "3": 77,
        "5": 11,
        "4": 22,
        "7": 1
      },
      "degree_dist": {
        "1": 3464,
        "3": 1034,
        "2": 10311,
        "4": 123,
        "6": 4,
        "5": 36,
        "7": 2
      }
    },
    "pred": {
      "indegree_dist": {
        "0": 3457,
        "1": 10527,
        "4": 85,
        "2": 629,
        "6": 55,
        "3": 141,
        "5": 45,
        "8": 14,
        "21": 1,
        "7": 17,
        "9": 1,
        "15": 1
      },
      "outdegree_dist": {
        "1": 11307,
        "0": 2990,
        "3": 117,
        "2": 365,
        "6": 35,
        "4": 80,
        "5": 53,
        "7": 18,
        "8": 6,
        "13": 1,
        "9": 1
      },
      "degree_dist": {
        "1": 3256,
        "2": 9030,
        "0": 1545,
        "7": 46,
        "4": 238,
        "6": 71,
        "3": 534,
        "5": 103,
        "8": 31,
        "9": 29,
        "12": 18,
        "34": 1,
        "11": 26,
        "10": 15,
        "15": 6,
        "13": 11,
        "14": 9,
        "16": 2,
        "17": 1,
        "22": 1
      }
    }
  },
  "parse_rate": 100.0
}

def merge_dicts(glob_pattern: str, keys_to_ignore: Set):
    """Reads all the dictionaries located at glob patter, 
    and creates a single dataframe where there's a row for each file,
    and a column for each key in the dictionary.

    Args:
        glob_pattern (str): Files from which you read.
        keys_to_ignore (Set): Keys which should not be added
    """
    all_dicts = dict()
    for file_path in glob(glob_pattern):
        with open(file_path, "r") as f:
            print(f"Reading {file_path}")
            dict_from_file = flatten_dict("", json.load(f), keys_to_ignore)
            just_file_name = os.path.basename(file_path).split(".")[0]
            all_dicts[just_file_name] = dict_from_file

    df = pd.DataFrame.from_dict(all_dicts, orient="index").reset_index()

    cols_to_keep = set()

    for col in set(df.columns):
      if ("ngram_stats" not in col) or ("distinct" in col):
        cols_to_keep.add(col)
    df = df[cols_to_keep]
    df.rename(columns={"index": "system"}, inplace=True)
    print(df.head())
    return df


def flatten_dict(top_level_key: str, input_dict, keys_to_ignore: Set):
    flattened_dict = dict()
    for key, value in input_dict.items():
        prefix = f"{top_level_key}_" if len(top_level_key) > 0 else ""
        if key in keys_to_ignore:
            continue
        if isinstance(value, dict):
            value = flatten_dict(key, value, keys_to_ignore)
            for sub_key, sub_value in value.items():
                flattened_dict[f"{prefix}{sub_key}"] = sub_value
        else:
            flattened_dict[f"{prefix}{key}"] = value
    return flattened_dict

def pprint_content_metrics(merged_dicts):
    tmp = merged_dicts[["system", "bleu", "rougeL", "pred_msttr_msttr-100_nopunct", "bleurt"]]
    for col in tmp:
      if col == "system":
        continue
      if col == "rougeL":
        tmp[col] = tmp[col].apply(lambda x: x * 100)
      tmp[col] = tmp[col].apply(lambda x: round(float(x), 2))
    print(tmp.head(100))



def pprint_struct_metrics(merged_dicts):
    tmp = merged_dicts[["system", "isomorphism", "ged_graph_edit_distance", "jsd_degree_jsd", "degree_stats_pred_avg_degree", "num_nodes_edges_pred_avg_num_nodes", "num_nodes_edges_pred_avg_num_edges"]]

    rename_dict = {
      "degree_stats_pred_avg_degree": "d",
      "num_nodes_edges_pred_avg_num_nodes": "|V|",
      "num_nodes_edges_pred_avg_num_edges": "|E|",
      "jsd_degree_jsd": "J",
      "ged_graph_edit_distance": "G",
      "isomorphism": "iso"
    }
    tmp.rename(columns=rename_dict, inplace=True)
    for col in tmp:
      if col == "system":
        continue
      if col == "rougeL":
        tmp[col] = tmp[col].apply(lambda x: x * 100)
      tmp[col] = tmp[col].apply(lambda x: round(float(x), 2))
    print(tmp.head(100))
    # pretty print each row
    col_order = ["system", "iso", "G", "J", "d", "|V|", "|E|"]
    for i, row in tmp.iterrows():
      output_vals = [f"{getattr(row, col)}" for col in col_order]
      print(",".join(output_vals))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob_pattern", type=str)
    parser.add_argument("--keys_to_ignore", type=str)
    parser.add_argument("--outputs", type=str)
    parser.add_argument("--struct", action="store_true")
    parser.add_argument("--content", action="store_true")
    args = parser.parse_args()
    keys_to_ignore = set(args.keys_to_ignore.split(","))
    merged_dicts = merge_dicts(args.glob_pattern, keys_to_ignore)
    merged_dicts.to_csv(args.outputs, index=False, sep="\t")
    if args.struct:
      pprint_struct_metrics(merged_dicts)
    elif args.content:
      pprint_content_metrics(merged_dicts)



