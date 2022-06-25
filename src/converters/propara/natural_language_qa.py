import json
import re
from collections import OrderedDict

from src.converters.propara.function_variable import CodePromptCreator
from src.prompting.constants import END, END_LINE
from typing import List, Union, Dict, Tuple

class NLPromptCreator(CodePromptCreator):
    def __init__(self, src_data=None):
        super().__init__(src_data, oracle_initial_state=False)

    @staticmethod
    def text_to_var_value(s):
        return {'-': 'does not exist', '?': 'is located at unknown place'}.get(s, f"is located at {s}")

    @staticmethod
    def var_value_to_text(s):
        return {'is located at unknown place': '?', 'does not exist': '-'}.get(s, s)

    def construct_prompt(self, goal: str, steps: List[str], states: Dict[str, Dict[str, str]]) -> str:
        prompt = []
        prompt.append("Document")
        comments = [f'step {i}: {step}' for i, step in enumerate(steps)]
        prompt += (comments)

        prompt.append('\n')
        prompt.append('List of participants:')
        for idx, (state_name, state_info) in enumerate(states.items()):
            prompt.append(f'participant {idx}: {state_name}')
        prompt = self.list_to_str(prompt)

        return prompt

    def extract_state_info_from_questions(self, questions):
        state_info = {}
        oracle = [OrderedDict() for _ in range(len(questions[0]['answers']))]
        state_name_idx_map = {}
        for idx, question in enumerate(questions):
            state_name = question['question'].split("the location/state of ")[1].strip()
            state_info[state_name] = {'init_value': self.text_to_var_value(question['answers'][0])}
            for step_idx, answer in enumerate(question['answers']):
                oracle[step_idx][state_name] = self.text_to_var_value(answer)
            state_name_idx_map[state_name] = idx
        return state_info, oracle, state_name_idx_map

    def add_step(self, prompt: str, step: str, step_idx: int, prediction: Dict[str, str] = None) -> str:
        new_prompt = []
        new_prompt.append("\n")
        new_prompt.append(f'Question {step_idx}: after step {step_idx} ({step.replace(".", "").lower()}) is executed where are the participants?')
        new_prompt.append("Answers:")
        new_prompt = self.list_to_str(new_prompt)
        new_prompt = f'{prompt}\n{new_prompt}'
        return new_prompt

    def add_prediction(self, prompt: str, prediction: Dict[str, str]) -> str:
        new_prompt = []

        # prev predictions
        for state_name, pred in prediction.items():
            new_prompt.append(f'{state_name} {pred}')

        new_prompt = self.list_to_str(new_prompt)
        new_prompt = f'{prompt}\n{new_prompt}'

        return new_prompt

    def generate_sample_head(self, sample):
        goal = sample.get('goal', 'main')
        steps = sample['steps']
        state_info, _, _ = self.extract_state_info_from_questions(sample['questions'])
        prompt_head = self.construct_prompt(goal, steps, state_info)
        return prompt_head

    def paragraph_to_code(self, sample):
        goal = sample.get('goal', 'main')
        steps = sample['steps']
        state_info, oracle, _ = self.extract_state_info_from_questions(sample['questions'])
        prompt = self.construct_prompt(goal, steps, state_info)
        prompt = self.add_step(prompt, steps[0], step_idx=0)

        for step_idx in range(1, len(steps)):
            step = steps[step_idx]
            prev_oracle = oracle[step_idx - 1]
            prev_oracle = {state_name: value for state_name, value in prev_oracle.items()}
            prompt = self.add_prediction(prompt, prev_oracle)
            prompt = self.add_step(prompt, step, step_idx)
        prev_oracle = {state_name: value for state_name, value in oracle[-1].items()}
        prompt = self.add_prediction(prompt, prev_oracle)

        return prompt

    def parse_line(self, line):
        if line.strip().endswith("does not exist"):
            state_name = line[:-len("does not exist")]
            value = "does not exist"
        elif line.strip().endswith("is located at unknown place"):
            state_name = line[:-len("is located at unknown place")]
            value = "is located at unknown place"
        elif "is located at" in line:
            try:
                state_name, value = line.split("is located at ")
            except ValueError:
                print(line)
                return None
        else:
            return None

        state_name = state_name.strip()
        value = value.strip()
        return state_name, value

    def code_to_predictions(self, sample, code):
        state_info, _, state_name_idx_map = self.extract_state_info_from_questions(sample['questions'])

        lines = code.split("\n")
        func_list = []
        func_body = None

        try:
            while not re.match('^Question \d+:', lines[0]):
                lines.pop(0)
            for line in lines:
                if '-----' in line:
                    break
                elif re.match(r'^Question \d+:', line):
                    if func_body is not None:
                        func_list.append(func_body)
                    func_body = [line.strip()]
                elif line.strip() != '':
                    func_body.append(line.strip())
            func_list.append(func_body)

        except IndexError:
            print("[ERROR] completely invalid output")


        predictions = [[None for _ in range(len(sample['steps']))] for _ in range(len(state_info))]
        for step_idx, func in enumerate(func_list):
            oracle_func_name = sample['steps'][step_idx].lower()
            overlap = [x for x in oracle_func_name.split() if x in func[0]]
            if len(overlap) / len(oracle_func_name.split()) < 0.5:
                print(f"[WARNING 1][{func[0]}] only covers part of oracle function name [{oracle_func_name}]")

            for line in func:
                parsed_line = self.parse_line(line)
                if parsed_line is None:
                    continue
                state_name, value = parsed_line
                value = self.var_value_to_text(value)
                state_idx = state_name_idx_map.get(state_name, 1000)
                if int(state_idx) < len(state_info):
                    predictions[int(state_idx)][step_idx] = value

        return predictions

if __name__ == "__main__":
    with open('data/propara/train.json') as f:
        data = json.load(f)
    converter = NLPromptCreator()
    # convert the whole sample
    prompt = converter.paragraph_to_code(data['7'])
    print(prompt)

    prediction = converter.code_to_predictions(data['7'], prompt)
    for i, p in enumerate(prediction):
        assert p == data['7']['questions'][i]['answers']

    prompt = converter.generate_sample_head(data['7'])
    print(prompt)