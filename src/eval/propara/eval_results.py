import argparse
import json
import random
import subprocess
from src.converters.propara.function_variable import CodePromptCreator
from src.converters.propara.class_attribute import CodePromptCreatorV2
from src.converters.propara.natural_language_qa import NLPromptCreator
from src.eval.propara.evaluator.evaluator import main as offical_eval

def eval(src_file, pred_file, save_file, job_type):
    with open(src_file, 'r') as f:
        src_d = json.load(f)

    with open(pred_file, 'r') as f:
        pred_d = {}
        for line in f:
            data = json.loads(line.strip())
            pred_d[str(data['id'])] = data

    if job_type == 'propara-func':
        converter = CodePromptCreator(src_data=src_d)
    elif job_type == 'propara-class':
        converter = CodePromptCreatorV2(src_data=src_d)
    elif job_type == 'propara-text-qa':
        converter = NLPromptCreator(src_data=src_d)
    else:
        raise NotImplementedError(job_type)

    for qid, src_data in src_d.items():
        pred = pred_d[qid]
        predictions = converter.code_to_predictions(src_data, pred['generated_graph'])
        converter.save_sample_prediction(qid, predictions)

    converter.save_predictions(save_file)

    # save json results to tsv results that propara official eval supports
    with open(save_file, 'r') as f_in, open(save_file.replace("json", "tsv"), 'w') as f_out:
        d = json.load(f_in)
        for qid, v in d.items():
            for state_id, pred_list in enumerate(v['predictions']):
                state_name = v['questions'][state_id]['question'][len("the location/state of "):]
                for step_idx in range(1, len(pred_list)):
                    prev = pred_list[step_idx - 1]
                    curr = pred_list[step_idx]
                    if curr is None or curr == '':
                        if prev:
                            print(f'[WARNING 2] Question {qid}: {state_name} at step {step_idx} is "{curr}" set to {prev}')
                            curr = prev
                        else:
                            print(f'[WARNING 3] Question {qid}: {state_name} at step {step_idx} and {step_idx - 1} are "{prev}" and "{curr}", set both to ?')
                            curr = '-'
                            prev = '-'
                    if prev == curr:
                        status = "NONE"
                    elif prev == '-' and curr != '-':
                        status = "CREATE"
                    elif prev != '-' and curr == '-':
                        status = "DESTROY"
                    else:
                        status = 'MOVE'
                    f_out.write(f'{qid}\t{step_idx}\t{state_name}\t{status}\t{prev}\t{curr}\n')


def sanity_check():
    with open('data/propara/predictions/function_variable_predictions.tsv', 'r') as f:
        lines = []
        for line in f:
            lines.append(line)

    random.shuffle(lines)
    with open('data/propara/predictions/function_variable_predictions.tmp.tsv', 'w') as f:
        for line in lines:
            f.write(line)

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_output_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--src_file', type=str, default='data/propara/test.json')
    parser.add_argument('--oracle_file', type=str, default='data/propara/test/answers.tsv')
    parser.add_argument('--job_type', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # sanity_check()
    args = config()
    save_file = args.output_file
    save_file_tsv = save_file.replace("json", "tsv")
    eval(args.src_file, args.raw_output_file, save_file, args.job_type)
    print(f"\n\nEval file {save_file_tsv} with oracle {args.oracle_file}")
    offical_eval(args.oracle_file, save_file_tsv, '', '', '')
