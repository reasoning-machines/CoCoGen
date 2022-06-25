# make dynamic prompt
import json
import sys
import logging
import pandas as pd
import argparse

from tqdm import tqdm
from prompting.knnprompt.examplestore import ExampleStore
from prompting.knnprompt.retriever import Retriever, SemanticMultiRetriever, SemanticRetriever

from src.prompting.constants import END

def _hacked_expl_key(row):
    key = f"Belief: {row['belief']} Argument: {row['argument']}"
    return key


def make_prompt(test_file_path: str, output_file_path: str, k: int,
                query_field: str,
                train_file_path: str = None,
                retrieval_model_name: str = None,
                checkpoint_path: str = None,
                retrieval_config_path: str = None,
                ):

    def _make_prompt(row):
        # makes KATE like prompt for a single example
        closest_queries, closest_documents = retriever.get_closest(
            query=row[query_field], k=k)
        prompt = ""
        # most relevant closest to the query
        for code in closest_documents[::-1]:
            prompt += code
            prompt += f"{END}\n\n"
        return closest_queries, prompt


    test_data = pd.read_json(test_file_path, orient='records', lines=True)
    test_data[query_field] = test_data.apply(_hacked_expl_key, axis=1)


    retriever = make_retriever(
        train_file_path, retrieval_model_name, checkpoint_path, retrieval_config_path, query_field=query_field)

    test_data_with_prompt_attached = []

    for i, row in tqdm(test_data.iterrows(), total=len(test_data)):
        tmp = row.copy()
        closest_queries, prompt = _make_prompt(row)
        tmp["prompt"] = prompt
        tmp["closest_queries"] = closest_queries
        test_data_with_prompt_attached.append(tmp)

    test_data_with_prompt_attached = pd.DataFrame(
        test_data_with_prompt_attached)
    test_data_with_prompt_attached.to_json(
        output_file_path, orient='records', lines=True)


def make_retriever(train_file_path: str, retrieval_model_name: str, checkpoint_path: str, retrieval_config_path: str, query_field: str) -> Retriever:

    def _make_single_retriever():
        example_store = make_example_store(
            train_file_path, retrieval_model_name, checkpoint_path=checkpoint_path, query_field=query_field)

        return SemanticRetriever(example_store)

    def _make_multi_retriever():
        with open(retrieval_config_path, "r") as f:
            config = json.load(f)

        examples_stores = []
        for config_item in config:
            if "checkpoint_path" in config_item:
                checkpoint_path = config_item["checkpoint_path"]
            else:
                checkpoint_path = None
            example_store = make_example_store(
                train_file_path, retrieval_model_name=config_item["retrieval_model_name"],
                checkpoint_path=checkpoint_path, query_field=query_field)
            examples_stores.append(example_store)
        return SemanticMultiRetriever(examples_stores)
    
    if retrieval_config_path is None:
        return _make_single_retriever()
    else:
        return _make_multi_retriever()


def make_example_store(train_file_path: str, retrieval_model_name: str, checkpoint_path: str, query_field: str) -> ExampleStore:
    examples = pd.read_json(train_file_path, orient='records', lines=True)
    examples[query_field] = examples.apply(_hacked_expl_key, axis=1)
    print(examples[query_field].head())
    examples = [(e[query_field], e["reference_code"])
                for _, e in examples.iterrows()]
    logging.info(f"Loaded {len(examples)} examples in the example store")

    return ExampleStore(retrieval_model_name=retrieval_model_name, examples=examples, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, required=True)
    parser.add_argument("--train_file_path", type=str, required=True)
    parser.add_argument("--retrieval_model_name", type=str,
                        default="all-mpnet-base-v2")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output_file_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--retrieval_config_path", type=str, required=False)
    parser.add_argument("--query_field", type=str, required=False, default="scenario")
    args = parser.parse_args()
    make_prompt(test_file_path=args.test_file_path,
                train_file_path=args.train_file_path,
                retrieval_model_name=args.retrieval_model_name,
                k=args.k,
                output_file_path=args.output_file_path,
                checkpoint_path=args.checkpoint_path,
                retrieval_config_path=args.retrieval_config_path,
                query_field=args.query_field)
