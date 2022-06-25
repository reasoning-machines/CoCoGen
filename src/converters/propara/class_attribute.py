import json

from src.converters.propara.function_variable import CodePromptCreator
from typing import List, Union, Dict, Tuple


class CodePromptCreatorV2(CodePromptCreator):
    def __init__(self, src_data=None):
        super().__init__(src_data, oracle_initial_state=False)

    @staticmethod
    def to_function_head(s):
        return f'def {s}(self):'

    def construct_prompt(self, goal: str, steps: List[str], states: Dict[str, Dict[str, str]]) -> str:
        prompt = []

        prompt.append('class Object:')
        prompt.append('\tdef __init__(self):')
        comments = [f'\t\t# {step}' for step in steps]
        prompt += (comments)

        # prompt.append('\n')
        # add state variables
        for idx, (state_name, state_info) in enumerate(states.items()):
            expl = state_info['explanation']
            prompt.append(f'\t\t# {expl}')
            prompt.append(f'\t\tself.{state_name} = None')

        prompt = self.list_to_str(prompt)
        return prompt


    def add_prediction(self, prompt: str, prediction: Dict[str, str]) -> str:
        new_prompt = []

        # prev predictions
        for state_name, pred in prediction.items():
            if pred not in ['None', 'True', 'False']:
                pred = f'"{pred}"'
            new_prompt.append(f'\t\tself.{state_name} = {pred}')

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
        # line 0 is class Object:
        for line in lines[1:]:
            if line.strip().startswith('def'):
                if func_body is not None:
                    func_list.append(func_body)
                func_body = [line.strip()]
            elif line.strip():
                func_body.append(line.strip())
        func_list.append(func_body)

        # remove main
        assert func_list[0][0].strip().startswith('def __init__'), func_list[0][0]
        func_list = func_list[1:]

        predictions = [[None for _ in range(len(sample['steps']))] for _ in range(len(state_info))]
        # predictions = [[None for _ in range(len(func_list))] for _ in range(len(state_info))]
        for step_idx, func in enumerate(func_list):
            oracle_func_name = self.to_function_name(sample['steps'][step_idx])
            overlap = [x for x in oracle_func_name.split("_") if x in func[0]]
            if len(overlap) / len(oracle_func_name.split("_")) < 0.5:
                print(f"[WARNING 1][{func[0]}] only covers part of oracle function name [{oracle_func_name}]")

            for line in func:
                if line.startswith("self.state_"):
                    tks = line.split('=')
                    state_name = tks[0][len("self."):]
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

        return predictions


if __name__ == "__main__":
    with open('data/propara/train.json') as f:
        data = json.load(f)
    converter = CodePromptCreatorV2()
    # convert the whole sample
    prompt = converter.paragraph_to_code(data['7'])
    print(prompt)

    prediction = converter.code_to_predictions(data['7'], prompt)
    for i, p in enumerate(prediction):
        assert p == data['7']['questions'][i]['answers']

    prompt = converter.generate_sample_head(data['7'])
    print(prompt)