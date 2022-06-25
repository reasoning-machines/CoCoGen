import argparse
import os
import time
from collections import defaultdict

import openai
import json
import sys
from typing import List, Union, Dict, Tuple

class CodePromptCreator:

    def __init__(self, src_data=None, oracle_initial_state=False):
        self.oracle_initial_state = oracle_initial_state
        self.src_data = src_data
        self.predictions = {}

    @staticmethod
    def to_function_name(s):
        s = s.replace(".", "").replace(",", "")
        # remove DT
        tok = s.lower().split()
        tok = [x for x in tok if x not in ['the', 'a', 'an']]
        return '_'.join(tok)

    @staticmethod
    def to_function_head(s):
        return f'def {s}():'


    @staticmethod
    def list_to_str(l):
        # remove \n
        l = [x.replace("\n", " ") if x != '\n' else '' for x in l]
        l = '\n'.join(l)
        return l

    @staticmethod
    def text_to_var_value(s):
        return {'-': 'None', '?': 'UNK'}.get(s, s)

    @staticmethod
    def var_value_to_text(s):
        return {'UNK': '?', 'None': '-'}.get(s, s)

    def construct_prompt(self, goal: str, steps: List[str], states: Dict[str, Dict[str, str]]) -> str:
        prompt = []

        func_head = self.to_function_head(self.to_function_name(goal))
        prompt.append(func_head)

        comments = [f'\t# {step}' for step in steps]
        prompt += (comments)

        # add state variables
        for state_name, state_info in states.items():
            expl = state_info['explanation']
            prompt.append(f'\t# {expl}')

        if self.oracle_initial_state:
            # state: state_i: explanation
            init_func_head = self.to_function_head(self.to_function_name('Init'))
            prompt.append(f"\t{init_func_head}")
            for state_name, state_info in states.items():
                init_value = state_info['init_value']
                if init_value not in ['None', 'True', 'False']:
                    init_value = f'"{init_value}"'
                cur_state = []
                cur_state.append(f"\t\t{state_name} = {init_value}")
                prompt += cur_state

        prompt = self.list_to_str(prompt)

        return prompt

    def add_prediction(self, prompt: str, prediction: Dict[str, str]) -> str:
        new_prompt = []

        # prev predictions
        for state_name, pred in prediction.items():
            if pred not in ['None', 'True', 'False']:
                pred = f'"{pred}"'
            new_prompt.append(f'\t\t{state_name} = {pred}')

        new_prompt = self.list_to_str(new_prompt)
        new_prompt = f'{prompt}\n{new_prompt}'

        return new_prompt

    def add_step(self, prompt: str, step: str, prediction: Dict[str, str]=None) -> str:
        new_prompt = []

        # step as function
        func_head = self.to_function_head(self.to_function_name(step))
        new_prompt.append(f'\t{func_head}')

        new_prompt = self.list_to_str(new_prompt)
        new_prompt = f'{prompt}\n{new_prompt}'

        return new_prompt

    def extract_state_info_from_questions(self, questions):
        state_info = {}
        oracle = [[None for _ in range(len(questions))] for _ in range(len(questions[0]['answers']))]
        for idx, question in enumerate(questions):
            state_name = f"state_{idx}"
            comment = f"{state_name} tracks {question['question'].lower()}"
            state_info[state_name] = {'explanation': comment, 'init_value': self.text_to_var_value(question['answers'][0])}
            for step_idx, answer in enumerate(question['answers']):
                oracle[step_idx][idx] = self.text_to_var_value(answer.strip())

        return state_info, oracle

    def paragraph_to_code(self, sample):
        goal = sample.get('goal', 'main')
        steps = sample['steps']
        state_info, oracle = self.extract_state_info_from_questions(sample['questions'])
        prompt = self.construct_prompt(goal, steps, state_info)
        prompt = self.add_step(prompt, steps[0])

        for step_idx in range(1, len(steps)):
            step = steps[step_idx]
            prev_oracle = oracle[step_idx - 1]
            prev_oracle = {f'state_{state_idx}': value for state_idx, value in enumerate(prev_oracle)}
            prompt = self.add_prediction(prompt, prev_oracle)
            prompt = self.add_step(prompt, step)
        prev_oracle = {f'state_{state_idx}': value for state_idx, value in enumerate(oracle[-1])}
        prompt = self.add_prediction(prompt, prev_oracle)

        return prompt

    def code_to_predictions(self, sample, code):
        state_info, _ = self.extract_state_info_from_questions(sample['questions'])

        lines = code.split("\n")
        func_list = []
        func_body = None
        for line in lines:
            if line.strip().startswith('def'):
                if func_body is not None:
                    func_list.append(func_body)
                func_body = [line.strip()]
            elif line.strip():
                func_body.append(line.strip())
        func_list.append(func_body)

        # remove main
        assert func_list[0][0].strip().startswith('def main')
        func_list = func_list[1:]

        predictions = [[None for _ in range(len(sample['steps']))] for _ in range(len(state_info))]
        # predictions = [[None for _ in range(len(func_list))] for _ in range(len(state_info))]
        for step_idx, func in enumerate(func_list):
            if step_idx >= len(sample['steps']):
                break
            oracle_func_name = self.to_function_name(sample['steps'][step_idx])
            oracle_func_name = oracle_func_name.replace('_', ' ')
            overlap = [x for x in oracle_func_name if x in func[0]]
            assert len(overlap) / len(oracle_func_name.split(("_"))) > 0.8, f"{func[0]} {oracle_func_name}"

            for line in func:
                if line.startswith("state_"):
                    tks = line.split('=')
                    state_name = tks[0]
                    state_name = state_name.strip()
                    state_idx = state_name.split("_")[1]
                    # check whether the line is valid
                    if len(tks) != 2 or not state_idx.isdigit():
                        print(f"Invalid line: {line}")
                        continue
                    value = tks[1]
                    value = self.var_value_to_text(value.strip().replace('"', ''))
                    if int(state_idx) < len(state_info):
                        predictions[int(state_idx)][step_idx] = value

        # # remove the placeholder for def main
        # for i, state_pred in enumerate(predictions):
        #     assert state_pred[0] is None
        #     state_pred = state_pred[1:]
        #     predictions[i] = state_pred

        return predictions

    def generate_sample_head(self, sample):
        goal = sample.get('goal', 'main')
        steps = sample['steps']
        state_info, _ = self.extract_state_info_from_questions(sample['questions'])
        prompt_head = self.construct_prompt(goal, steps, state_info)
        return prompt_head

    def save_sample_prediction(self, question_id, prediction):
        self.predictions[question_id] = prediction

    def save_predictions(self, save_file):
        for k, v in self.src_data.items():
            v['predictions'] = self.predictions[k]
        with open(save_file, 'w') as f:
            json.dump(self.src_data, f, indent=2)

if __name__ == "__main__":
    with open('data/propara/train.json') as f:
        data = json.load(f)
    converter = CodePromptCreator(oracle_initial_state=False)
    # convert the whole sample
    prompt = converter.paragraph_to_code(data['7'])
    print(prompt)

    # we have to provide the init state to the sample
    prompt = converter.generate_sample_head(data['7'])
    print(prompt)

    code = """def main():
	# Init
	# Magma rises from deep in the earth.
	# The magma goes into volcanos.
	# The volcanos pressure the magma upwards.
	# The pressure causes the magma to push through the surface of the volcano.
	# The lava cools.
	# The lava forms new rock.
	# New magma is pressured to the surface of the volcano.
	# The volcano bursts through the rock the formed after the last eruption.
	# state_0 tracks the location/state of lava
	# state_1 tracks the location/state of magma
	# state_2 tracks the location/state of new rock
	def init():
		state_0 = None
		state_1 = "deep in the earth"
		state_2 = None
	def magma_rises_from_deep_in_earth():
		state_0 = None
		state_1 = "deep in the earth"
		state_2 = None
	def magma_goes_into_volcanos():
		state_0 = None
		state_1 = "volcano"
		state_2 = None
	def volcanos_pressure_magma_upwards():
		state_0 = None
		state_1 = "volcano"
		state_2 = None
	def pressure_causes_magma_to_push_through_surface_of_volcano():
		state_0 = "UNK"
		state_1 = None
		state_2 = None
	def lava_cools():
		state_0 = "UNK"
		state_1 = None
		state_2 = None
	def lava_forms_new_rock():
		state_0 = None
		state_1 = None
		state_2 = "UNK"
	def new_magma_is_pressured_to_surface_of_volcano():
		state_0 = None
		state_1 = None
		state_2 = "UNK"
	def volcano_bursts_through_rock_formed_after_last_eruption():
		state_0 = None
		state_1 = None
		state_2 = "UNK"
    """
    # conver the prediction to the answers
    predictions = converter.code_to_predictions(data['7'], code)
    print(json.dumps(predictions, indent=2))

    # save the all the predictions once finish
    # converter.save_predictions('data/propara/predictions/function_variable_predictions.json')





