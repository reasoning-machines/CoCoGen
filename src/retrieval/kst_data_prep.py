import pickle
import logging
import pathlib
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from nltk.translate.bleu_score import sentence_bleu
random.seed(42)

def _make_event_str(events_dict):
    num_events = len(events_dict)
    event_str = []
    for i in range(num_events):
        e = events_dict[str(i)].lower().strip()
        if e[-1] != ".":
            e += "."
        event_str.append(e)
    return " ".join(event_str)

def make_pairs(inpath: str, outpath: str):


    model = SentenceTransformer("all-mpnet-base-v2").cuda()
    data = pd.read_json(inpath, lines=True, orient="records")
    data["event_str"] = data["events"].apply(_make_event_str)
    encodings = model.encode(data["event_str"].tolist())


    
    # iterate over all pairs of rows in data
    for i, row_i in tqdm(data.iterrows(), total=len(data), desc="Making paired training data"):
        pairs = []
        # enumerate from i + 1
        for j in range(i + 1, len(data)):
            row_j = data.iloc[j]
            events_sim_score, e1, e2 = get_events_sim(row_i, row_j)
            sem_sim_score = util.cos_sim(encodings[i], encodings[j]).item()
            tmp = {
                "idx_1": i,
                "idx_2": j,
                "events_sim_score": events_sim_score,
                "scenario_1": row_i.scenario,
                "scenario_2": row_j.scenario,
                "scenario_1_events": e1,
                "scenario_2_events": e2,
                "scenario_1_meta": row_i,
                "scenario_2_meta": row_j,
                "sem_sim_score": sem_sim_score
            }
            pairs.append(tmp)
            # also add example with swapped indices
            pairs.append({
                "idx_1": j,
                "idx_2": i,
                "events_sim_score": events_sim_score,
                "scenario_1": row_j.scenario,
                "scenario_2": row_i.scenario,
                "scenario_1_events": "",
                "scenario_2_events": "",
                "scenario_1_meta": dict(),
                "scenario_2_meta": dict(),
                "sem_sim_score": sem_sim_score
            })

        paired_data = pd.DataFrame(pairs)
        paired_data.to_json(f"{outpath}/{i}.jsonl", orient="records", lines=True)
    # z = (paired_data["events_sim_score"].max() - paired_data["events_sim_score"].min())
    # paired_data["events_sim_score_normed"] = (
    #     paired_data["events_sim_score"] - paired_data["events_sim_score"].min()) / z
    # print(paired_data["events_sim_score_normed"].describe())

    # z = (paired_data["sem_sim_score"].max() - paired_data["sem_sim_score"].min())
    # paired_data["sem_sim_score_normed"] = (
    #     paired_data["sem_sim_score"] - paired_data["sem_sim_score"].min()) / z
    # print(paired_data["sem_sim_score_normed"].describe())


    # train, test = train_test_split(paired_data, test_size=0.03)
    # dev, test = train_test_split(test, test_size=0.5)
    # train.to_json(f"{outpath}/train.jsonl", orient="records", lines=True)
    # dev.to_json(f"{outpath}/dev.jsonl", orient="records", lines=True)
    # test.to_json(f"{outpath}/test.jsonl", orient="records", lines=True)


def get_events_sim(row_1: dict, row_2: dict) -> str:


    rows1_events = row_1.event_str
    rows2_events = row_2.event_str

    # calculate BLEU score
    
    
    return sentence_bleu([rows1_events], rows2_events), rows1_events, rows2_events


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, required=True)
    parser.add_argument("--outpath", type=str, required=True)

    args = parser.parse_args()

    make_pairs(
        inpath=args.inpath, outpath=args.outpath)
