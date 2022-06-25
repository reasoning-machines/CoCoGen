from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd

def run_eval(model, test_data):
    grouped_data = test_data.groupby("sentence1")
    best_match_edist = []
    i = 0
    for sentence1, group in grouped_data:
        if i > 10:
            break
        best_match = get_best_match(model, sentence1, group)
        best_match_edist.append(best_match["edist"])
        i += 1
    return np.array(best_match_edist).mean()


def get_best_match(model, sentence1, candidates):
    sentence1_embedding = model.encode([sentence1])
    candidate_embeddings = model.encode(candidates["sentence2"].tolist())
    ir_score = util.pytorch_cos_sim(sentence1_embedding, candidate_embeddings)
    best_match = candidates.iloc[ir_score.argmax().item()]
    return best_match


def get_sim_examples(model, train_data_path: str, dev_data_path: str):
    train_data = pd.read_json(train_data_path, lines=True, orient="records")
    dev_data = pd.read_json(dev_data_path, lines=True, orient="records")
    train_data_scenarios = train_data["scenario"].to_list()
    dev_data_scenarios = dev_data["scenario"].to_list()
    train_scenario_embeddings = model.encode(train_data_scenarios)
    dev_scenario_embeddings = model.encode(dev_data_scenarios)
    results = []
    for i, row in dev_data.iterrows():
        ir_scores = util.pytorch_cos_sim(dev_scenario_embeddings[i], train_scenario_embeddings)
        # take top 5 structures
        top_5_idxs = ir_scores.argsort()[-5:][::-1]
        tmp = {
            "dev_example": row,
            "top_5_train_examples": [train_data.iloc[idx].to_dict() for idx in top_5_idxs],
        }
        results.append(tmp)
    return results

