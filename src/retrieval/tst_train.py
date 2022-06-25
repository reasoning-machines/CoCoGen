import pickle
from pprint import pprint
import torch
from torch import nn, Tensor
from typing import Iterable, Dict, List
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    evaluation,
    util
)
from tqdm import tqdm
from torch import nn
import torch
from torch.utils.data import DataLoader
import pandas as pd
import logging

from src.eval.graph_metrics import NxMetrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", level=logging.INFO
)


def run(args):
    model = get_model(args.model_name)
    train(
        model,
        train_path=args.train_path,
        test_path=args.test_path,
        out_path=args.out_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        label_col=args.label_col,
        lr=args.lr,
        args=args
    )


def get_model(model_name: str) -> SentenceTransformer:
    logging.info("Initializing model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name).to(device)
    return model


def train(
    model: SentenceTransformer,
    train_path: str,
    test_path: str,
    out_path: str,
    label_col: str,
    lr: float,
    batch_size: int,
    epochs: int,
    args: dict
):

    train_examples = read_examples(train_path, label_col)
    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size)
    train_loss = StructSimLoss(model)

    test_examples = read_examples(test_path, label_col)

    evaluator = evaluation.SequentialEvaluator([EdistEvaluator(model=model, base_model_name=args.model_name,
                                                               train_graphs_path=args.train_graphs_path,
                                                               eval_graphs_path=args.eval_graphs_path),
                                                MSEEvaluator(
                                                    model=model, test_examples=test_examples)
                                                ])

    total_n_steps = (len(train_examples) * epochs) // args.batch_size
    warmup_steps = int(total_n_steps * 0.2)
    evaluation_steps = int(total_n_steps * 0.1)
    logging.info(f"Total number of steps: {total_n_steps}")
    logging.info(f"Warmup steps: {warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluator=evaluator,
        evaluation_steps=evaluation_steps,
        optimizer_params={"lr": lr}
    )

    # save model
    torch.save(model.state_dict(), out_path)


def read_examples(path: str, label_col: str = "struct_sim") -> List[InputExample]:
    examples_df = pd.read_json(path, orient="records", lines=True)
    examples = []
    for _, row in tqdm(examples_df.iterrows(), total=len(examples_df), desc="Reading examples"):
        label = float(row[label_col])
        examples.append(
            InputExample(
                texts=[row["sentence1"], row["sentence2"]],
                label=label,
            )
        )
    return examples


class StructSimLoss(nn.Module):
    # adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CosineSimilarityLoss.py
    def __init__(
        self,
        model: SentenceTransformer,
        loss_fct=nn.MSELoss(reduction="none"),
        cos_score_transformation=nn.Identity(),
    ):
        super(StructSimLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        """Custom loss function:
        loss = (sentence_cosine - struct_sim)^2
        Returns:
            _type_: _description_
        """
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]
        sentence_similarity = self.cos_score_transformation(
            torch.cosine_similarity(embeddings[0], embeddings[1])
        )

        struct_sim_loss = self.loss_fct(
            sentence_similarity, labels.view(-1)).mean()
        return struct_sim_loss


class MSEEvaluator(evaluation.SentenceEvaluator):
    def __init__(
        self, model, test_examples, bsz: int = 32
    ) -> None:
        self.test_examples = test_examples
        self.labels = torch.Tensor(
            [eg.label for eg in self.test_examples]).to(model.device)
        self.bsz = bsz
        self.num_examples = len(self.test_examples)
        self.loss_fct = nn.MSELoss(reduction="none")
        super().__init__()

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        self.sentence1 = model.encode(
            [eg.texts[0] for eg in self.test_examples],
            show_progress_bar=True,
            convert_to_tensor=True,
            device=model.device,
        )
        self.sentence2 = model.encode(
            [eg.texts[1] for eg in self.test_examples],
            show_progress_bar=True,
            convert_to_tensor=True,
            device=model.device,
        )

        total_mse = 0.0
        for i in range(0, self.num_examples, self.bsz):
            loss = self.calc_mse(
                sentence1_embedding=self.sentence1[i: i + self.bsz, :],
                sentence2_embedding=self.sentence2[i: i + self.bsz, :],
                labels=self.labels[i: i + self.bsz],
            )
            total_mse += loss.item()
        print()
        print("-" * 80)
        print(f"MSE: {total_mse / self.num_examples:.4f}")
        print("-" * 80)
        print()
        return -total_mse / self.num_examples

    def calc_mse(
        self, sentence1_embedding, sentence2_embedding,  labels
    ):
        sentence_similarity = torch.cosine_similarity(
            sentence1_embedding, sentence2_embedding)
        return self.loss_fct(sentence_similarity, labels).sum()


class EdistEvaluator(evaluation.SentenceEvaluator):
    def __init__(
        self, model, train_graphs_path: str,
        base_model_name: str,
        eval_graphs_path: str,
        n_task_samples: int = 300
    ) -> None:
        logging.info("Creating edist evaluator")
        with open(train_graphs_path, "rb") as f:
            self.task_train_data = pickle.load(f)
        with open(eval_graphs_path, "rb") as f:
            self.task_eval_data = pickle.load(f)
            self.task_eval_data = self.task_eval_data[:n_task_samples]

        logging.info(f"Loaded {len(self.task_train_data)} train graphs")

        base_model = SentenceTransformer(base_model_name).to(
            "cuda:1" if torch.cuda.is_available() else "cpu")

        self.edist_cache = dict()
        self.base_model_edist = self.calc_base_edist(base_model)
        logging.info(f"{base_model_name} edist = {self.base_model_edist}")
        # don't need the base model anymore

        self.switch = 0
        del base_model
        super().__init__()

    def calc_base_edist(self, model, k: int = 3):
        
        train_scenarios = model.encode(
            [eg["scenario"].lower() for eg in self.task_train_data])
        eval_scenarios = model.encode(
            [eg["scenario"].lower() for eg in self.task_eval_data])
        total_edist = 0.0
        total_pairs = 0
        for eval_idx in tqdm(range(len(eval_scenarios)), desc="Calculating edit distance for eval", total=len(eval_scenarios)):
            scores = util.cos_sim(
                eval_scenarios[eval_idx], train_scenarios).squeeze(0)
            # get closest 3 train scenarios
            top_k_idx = torch.topk(scores, k=min(k, len(scores)), dim=0)[1]

            closest_nx_graphs = []
            closest_nx_graph_scenarios = []
            for i in top_k_idx:
                scenario_a = self.task_train_data[i.item()]["scenario"].lower()
                scenario_b = self.task_eval_data[eval_idx]["scenario"].lower()
                key = sorted([scenario_a, scenario_b])
                key = "_".join(key)
                if key in self.edist_cache:
                    total_edist += self.edist_cache[key]
                else:
                    closest_nx_graphs.append(
                        self.task_train_data[i.item()]["nx_graph"])
                    closest_nx_graph_scenarios.append(scenario_a)

            if len(closest_nx_graphs) > 0:
                avg_edist, edit_dist_arr = NxMetrics.graph_edist(
                    closest_nx_graphs, [self.task_eval_data[eval_idx]["nx_graph"] for _ in range(len(closest_nx_graphs))], timeout=2, verbose=False, return_arr=True)

                for i in range(len(closest_nx_graphs)):
                    key = sorted([closest_nx_graph_scenarios[i],
                                self.task_eval_data[eval_idx]["scenario"].lower()])
                    key = "_".join(key)
                    self.edist_cache[key] = edit_dist_arr[i]
                    total_edist += edit_dist_arr[i]
            total_pairs += len(top_k_idx)
        return total_edist / total_pairs

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if self.switch % 2 == 0:
            self.switch += 1
        else:
            self.switch += 1
            return self.curr_edist
        self.curr_edist = self.calc_base_edist(model)
        print("-" * 80)
        print(
            f"EDIST: {self.curr_edist:.4f}, baseline: {self.base_model_edist:.4f}")
        improvement = (self.base_model_edist - self.curr_edist) * \
            100 / self.base_model_edist
        print(f"Improvement: {improvement:.4f}%")
        print("-" * 80)

def norm(edist_data):
    # z = (edist_data["edist"].max() - edist_data["edist"].min())
    # edist_data["edist_normed"] = (edist_data["edist"] - edist_data["edist"].min()) / z
    edist_data["struct_sim"] = edist_data["edist"].apply(lambda x: 1 if x == 0 else 0)
    return edist_data



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="all-mpnet-base-v2"
    )
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--label_col", type=str, default="struct_sim")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_graphs_path", type=str)
    parser.add_argument("--eval_graphs_path", type=str)
    args = parser.parse_args()
    run(args)
