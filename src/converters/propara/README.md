# Basic
## Convert a sample to the code
```python
with open('data/propara/train.json') as f:
    data = json.load(f)
converter = CodePromptCreator()
prompt = converter.paragraph_to_code(data['7'])
print(prompt)
```

## Convert the initial state to the code
We have to provide the initial state of the sample.
```python
prompt = converter.generate_sample_head(data['7'])
print(prompt)
```

## Conver the code to the answers
```python
code = """def magma_rises_from_deep_in_earth():
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
predictions = converter.code_to_predictions(data['7'], code)
print(json.dumps(predictions, indent=2))
```

## For each prediction, save it
```python
converter.save_sampple_prediction('7', predictions)
```

## Post-processing, save the all once finish
```python
converter.save_predictions('data/propara/predictions/function_variable_predictions.json')
```