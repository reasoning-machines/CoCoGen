# Language Models of Code are Few-Shot Commonsense Learners

The official repository for "Language Models of Code are Few-Shot Commonsense Learners" (Madaan et al., EMNLP'2022).

This paper addresses the general task of *structured* commonsense reasoning: 
generate a *graph* given a natural language input.
We address these family of tasks by framing the problem as a **code generation** task, and prompting large language models of **code** (e.g., Codex).

| <img alt="image" width="70%" src="https://user-images.githubusercontent.com/15002544/194352035-f0ae52f9-8fe3-47f7-af2e-2c1dc2461ad2.png"> |
|-|

| <img width="70%" alt="image" src="https://user-images.githubusercontent.com/15002544/194354502-460dcda7-b497-44b2-bdca-75240cb0664c.png"> |
|-|

## Table of contents

- [Running CoCoGen](#running-cocogen)
- [Sample Outputs](#sample-outputs)
- [End-to-end workflow](#end-to-end-workflow)
- [Creating dynamic prompts (KST)](#creating-dynamic-prompts-kst)


## Running CoCoGen

An OpenAI API key is required to run the jobs. To get an API key, register at [https://openai.com/blog/openai-codex/](https://openai.com/blog/openai-codex/).

The key should be exported in the environment variable `OPENAI_API_KEY`.

Please note that as of Oct 2022, codex is free to use for non-commercial purposes with a key.



1. Proscript script generation:

```
python -u src/api/query_openai_over_tasks.py --task_file_path data/proscript_script_gen/dev.jsonl --num_tasks -1 \
                                                 --output_file_path data/proscript_script_gen/dev_outputs.jsonl \
                                                 --prompt_path  data/proscript_script_gen/prompt.txt --job_type proscript-literal \
                                                  --engine text-davinci-002 --max_requests_per_min 10
```


Here:

    * `--task_file_path`: path to the file containing the tasks to be run
    * `--num_tasks`: number of tasks to be run. If -1, all tasks in the file will be run.
    * `--output_file_path`: path to the file to which the outputs will be written
    * `--prompt_path`: path to the file containing the prompt. The prompt is created from the train split.
    * `--job_type`: the type of job to run. This is used to determine the code converter. Currently, the following converters are supported: "explagraphs",         "proscript-hashmap-init", "proscript-hashmap-sep", "proscript-networkx", "proscript-literal", "proscript-oo"

    * `--engine`: the API engine to use. See the complete list of available at https://beta.openai.com/docs/api-reference/introduction.



2. Proscript edge prediction:

```
python -u src/api/query_openai_over_tasks.py --task_file_path data/proscript_edge_prediction/dev.jsonl --num_tasks -1 \
                                                 --output_file_path data/proscript_edge_prediction/dev_outputs.jsonl \
                                                 --prompt_path  data/proscript_edge_prediction/prompt.txt --job_type "proscript-edge-pred-literal" \
                                                  --engine text-davinci-002 --max_requests_per_min 10
```


* Evaluation:
```sh
python src/eval/edge_pred_eval.py  data/proscript_edge_prediction/dev_outputs.jsonl
```

3. Explagraphs:

```
python -u src/api/query_openai_over_tasks.py --task_file_path data/explagraphs/dev.jsonl --num_tasks -1 \
                                                 --output_file_path data/explagraphs//dev_outputs.jsonl \
                                                 --prompt_path  data/explagraphs//prompt.txt --job_type "explagraphs-literal" \
                                                  --engine text-davinci-002 --max_requests_per_min 10
```


* Evaluate:
```sh
export CUDA_VISIBLE_DEVICES=0,1 && python src/eval/explagraph_eval.py data/explagraphs//dev_outputs.jsonl tmp
```

4. Propara:

```
python -u src/api/query_openai_over_tasks.py --task_file_path data/propara/test.jsonl --num_tasks -1 \
                                                 --output_file_path data/propara//test_outputs.jsonl \
                                                 --prompt_path  data/propara//prompt.txt --job_type "propara-func" \
                                                  --engine text-davinci-002 --max_requests_per_min 10
```

* Evaluate:
```sh
python src/eval/propara/eval_results.py --raw_output_file data/propara//test_outputs.jsonl --output_file  data/propara/test_output_predictions.jsonl
```


---

## Sample Outputs

Sample outputs for each task are located in `outputs`.

```
outputs/
├── explagraphs
├── propara
├── proscript_edge_pred
└── proscript_script_generation
```

 Since we are not permitted by the authors of proscript to release the test split, we remove the reference outputs for proscript_script_generation and proscript_edge_prediction.



### Output format
Each output file is a jsonl, where each line is a json object with several common fields:

```js
{

    "prompt": "the dynamic prompt created for the task"",
    "reference_graph":  "the graph/table for the task (G in paper),
    "reference_code": "python code for the graph (T + G_c in paper))",
    "codex_response": { 
        response from codex
    },
    "generated_code": "code generated by codex",
    "generated_graph": {
        same format as reference_graph, obtained by parsing the generated code
    }
}
```

For example, for edge prediction:

```js
{
    "scenario": "have bowl for cut ingredients",
    "closest_queries": [
        "mix ingredients together",
        "add the measured ingredients to bowl",
        "take out a cutting board and knife",
        "gather the ingredients",
        "measure out ingredients",
        "put all ingredients into bowl and mix",
        "prepare the ingredients with a knife",
        "make a list of ingredients needed",
        "copy list from website to paper",
        "put dry ingredients in one bowl",
        "prepare a meal"
    ],
    "prompt": "the dynamic prompt created for the task"",
        "reference_graph": {
        "title": "have bowl for cut ingredients",
        "num_steps": 7,
        "schema": [
            "step0: Move toward where work is to be done",
            "step1: Find the location of desired bowl",
            "step2: Grab the bowl firmly",
            "step3: Set the bowl down",
            "step4: Walk toward the bowls location",
            "step5: get out ingredients",
            "step6: have bowl for cut ingredients"
        ],
        "relations": [
            "step1 -> step4",
            "step4 -> step2",
            "step2 -> step0",
            "step0 -> step3",
            "step3 -> step6",
            "step5 -> step1"
        ]
    },
    "reference_code": "class HaveBowlForCutIngredients:\n\n    title = \"have bowl for cut ingredients\"\n    steps = 7\n\n    def step0(self):\n        return \"Move toward where work is to be done\"\n\n    def step1(self):\n        return \"Find the location of desired bowl\"\n\n    def step2(self):\n        return \"Grab the bowl firmly\"\n\n    def step3(self):\n        return \"Set the bowl down\"\n\n    def step4(self):\n        return \"Walk toward the bowls location\"\n\n    def step5(self):\n        return \"get out ingredients\"\n\n    def step6(self):\n        return \"have bowl for cut ingredients\"\n\n    def get_relations(self):\n        return [\n            \"step1 -> step4\",\n            \"step4 -> step2\",\n            \"step2 -> step0\",\n            \"step0 -> step3\",\n            \"step3 -> step6\",\n            \"step5 -> step1\",\n        ]\n",
    "input_prompt_code": "class HaveBowlForCutIngredients:\n\n    title = \"have bowl for cut ingredients\"\n    steps = 7\n\n    def step0(self):\n        return \"Move toward where work is to be done\"\n\n    def step1(self):\n        return \"Find the location of desired bowl\"\n\n    def step2(self):\n        return \"Grab the bowl firmly\"\n\n    def step3(self):\n        return \"Set the bowl down\"\n\n    def step4(self):\n        return \"Walk toward the bowls location\"\n\n    def step5(self):\n        return \"get out ingredients\"\n\n    def step6(self):\n        return \"have bowl for cut ingredients\"\n\n",
    "codex_response": {
        "id": "",
        "object": "text_completion",
        "created": ,
        "model": "code-davinci:002",
        "choices": [
            {
                "text": "    def get_relations(self):\n        return [\n            \"step4 -> step1\",\n            \"step1 -> step2\",\n            \"step2 -> step3\",\n            \"step3 -> step6\",\n            \"step5 -> step0\",\n            \"step5 -> step4\",\n        ]\n",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }
        ]
    },
    "generated_code": "class HaveBowlForCutIngredients:\n\n    title = \"have bowl for cut ingredients\"\n    steps = 7\n\n    def step0(self):\n        return \"Move toward where work is to be done\"\n\n    def step1(self):\n        return \"Find the location of desired bowl\"\n\n    def step2(self):\n        return \"Grab the bowl firmly\"\n\n    def step3(self):\n        return \"Set the bowl down\"\n\n    def step4(self):\n        return \"Walk toward the bowls location\"\n\n    def step5(self):\n        return \"get out ingredients\"\n\n    def step6(self):\n        return \"have bowl for cut ingredients\"\n\n    def get_relations(self):\n        return [\n            \"step4 -> step1\",\n            \"step1 -> step2\",\n            \"step2 -> step3\",\n            \"step3 -> step6\",\n            \"step5 -> step0\",\n            \"step5 -> step4\",\n        ]\n",
    "generated_graph": {
        "title": "have bowl for cut ingredients",
        "num_steps": 7,
        "schema": [
            "step0: Move toward where work is to be done",
            "step1: Find the location of desired bowl",
            "step2: Grab the bowl firmly",
            "step3: Set the bowl down",
            "step4: Walk toward the bowls location",
            "step5: get out ingredients",
            "step6: have bowl for cut ingredients"
        ],
        "relations": [
            "step4 -> step1",
            "step1 -> step2",
            "step2 -> step3",
            "step3 -> step6",
            "step5 -> step0",
            "step5 -> step4"
        ]
    },
    "elapsed_time": 0.0003092289
}
```


---


## End-to-end workflow


- To give an overview of our end-to-end workflow, we provide all the files required to run propara end-to-end.


#### Step 1: Create task files


```sh
python src/prompting/make_task_file.py --inpath data/propara/train.json --outpath data/propara/code/func/train.jsonl --job_type "propara-func"

python src/prompting/make_task_file.py --inpath data/propara/test.json --outpath data/propara/code/func/test.jsonl --job_type "propara-func"

python src/prompting/make_task_file.py --inpath data/propara/dev.json --outpath data/propara/code/func/dev.jsonl --job_type "propara-func"
```


#### Step 2: Create the prompt

```sh
python src/prompting/make_codex_prompt.py data/propara/code/func/train.jsonl 6 > data/propara/code/func/prompt.txt && cat data/propara/code/func/prompt.txt|it2copy
```

#### Step 3: Run inference


```sh
python -u src/codexapi/query_openai_over_tasks.py --task_file_path data/propara/code/func/test.jsonl --num_tasks -1 --output_file_path data/propara/code/func/test_output.jsonl --prompt_path data/propara/code/func/prompt.txt --job_type "propara-func" --engine code-davinci-002 --max_requests_per_min 10  --max_tokens 800
```

* The max tokens is set to 800 (vs. 280 default) because the generations are long. The prompt fits about 5 examples.

#### Step 4: Evaluate

```sh
python src/eval/propara/eval_results.py --raw_output_file data/propara/code/func/test_output.jsonl --output_file  data/propara/predictions/test_output.json 
```

Evaluate `data/propara/code/func/test_output.jsonl`


---

## Creating dynamic prompts

- Note that when we run a job with a fixed random prompt, the `prompt_path` is supplied as an argument to `query_openai_over_tasks.py`. In case of dynamic prompts, we will create a prompt for *each test example*, and store it in the task jsonl file with each example in the `prompt` field.

- In summary, we will call `query_openai_over_tasks.py` as before, but using a different task file and without specifying a prompt_path.

- To create this file, we use the following command:

```sh
python src/prompting/knnprompt/make_knn_like_prompt.py\
       --train_file_path ${TRAIN_EXAMPLES_PATH}\
       --test_file_path ${TEST_FILE_PATH} \
       --k $K 
       --output_file_path ${KNN_TEST_FILE_PATH} \
       --query_field ${QFIELD}
```

Here:
* `TRAIN_EXAMPLES_PATH` is the path to the training examples used to create the prompt. Note that this file should be the same file that is used for creating the prompts (it's the file we get after running `make_task_file`).

* `TEST_FILE_PATH` is the path to the test file. A prompt will be created for each example in this file, and written to the output file ${KNN_TEST_FILE_PATH}.

* `K` is the number of examples (nearest neighbors) to use for creating the prompt.

* `QUERY_FIELD` is the field in the test file that will be used to do knn search.


By default, we use "all-mpnet-base-v2" as our similarity model, but it can be changed using `--retrieval_model_name`


* An example with explagraph is:

```sh
python src/prompting/knnprompt/make_knn_like_prompt.py --train_file_path data/explagraphs/train.jsonl --test_file_path data/explagraphs/dev.jsonl --k 21 --output_file_path data/explagraphs/dev_kst_task.jsonl  --query_field belief
```

Once this command finishes, we can query codex as usual using the following:

```sh
python -u src/codexapi/query_openai_over_tasks.py --task_file_path data/explagraphs/dev_kst_task.jsonl --num_tasks -1 --output_file_path data/explagraphs/dev_kst_task_OUTPUT.jsonl  --job_type explagraph-relation --engine code-davinci-002

```

Note that no prompt path is specified, as the task file `data/explagraphs/dev_kst_task.jsonl` contains a prompt for each example.




