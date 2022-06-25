import logging
from typing import List
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)

class ExampleStore(object):
    """Stores examples for creating in-context prompts"""
    def __init__(
        self,
        retrieval_model_name: str,  # the name of the retrieval model to use e.g. bert-base-nli-stsb-mean-tokens
        examples: List,  # list of key value pairs (query, answer)
        checkpoint_path: str = None  # path to the checkpoint to load
    ):
        self.retrieval_model_name = retrieval_model_name
        self.checkpoint_path = checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.examples = self.deduplicate_examples(examples)
        self.example_to_idx = {example: idx for idx, example in enumerate(self.examples)}

        self.create_model()
        self.create_query_embeddings()


    def create_model(self):
        logging.info("Step 1: Initializing model...")
        self.model = SentenceTransformer(self.retrieval_model_name).to(self.device)
        logging.info("Model initialized")
        if self.checkpoint_path is not None:
            logging.info(f"Loading model checkpoint from {self.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            logging.info("Checkpoint loaded")

    def create_query_embeddings(self):
        logging.info("Step 2: Creating query embeddings!")
        queries = [example[0].lower() for example in self.examples]
        self.query_embeddings = self.model.encode(queries, batch_size=64, device=self.device)

        logging.info("Query embeddings created")

        self.query_embedding_cache = dict()

    def encode_query(self, query: str):
        query = query.strip().lower()
        if query in self.query_embedding_cache:
            query_embedding = self.query_embedding_cache[query]
        else:
            query_embedding = self.model.encode(query, show_progress_bar=False)
            self.query_embedding_cache[query] = query_embedding
        return query_embedding

    
    def deduplicate_examples(self, examples: List):
        seen = set()
        new_examples = []
        for example in examples:
            scenario = example[0].strip().lower()
            if scenario[-1] == ".":
                scenario = scenario[:-1]
            if scenario not in seen:
                new_examples.append((scenario, example[1]))
                seen.add(scenario)
        return new_examples
