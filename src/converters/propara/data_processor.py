import json
from collections import defaultdict


def convert_data(para_file, answer_file, save_file):
    INVALID_DATA = [263, 893, 901, 1151]
    d = defaultdict(lambda: {'steps': {},
                             'questions': defaultdict(lambda: {'question': '', 'answers': {}})})

    with open(para_file, 'r') as f:
        for line in f:
            tks = line.strip().split("\t")
            para_id, step_id, sent = tks
            para_id = int(para_id)
            step_id = int(step_id)
            d[para_id]['steps'][step_id] = sent
        for k, v in d.items():
            # add init step
            v['steps'][0] = 'Init'

    object_id_map = defaultdict(dict)
    with open(answer_file, 'r') as f:
        for line in f:
            tks = line.strip().split("\t")

            para_id, step_id, object, action, before, after = tks
            para_id = int(para_id)
            step_id = int(step_id)

            if object in object_id_map[para_id]:
                object_id = object_id_map[para_id][object]
            else:
                object_id = len(object_id_map[para_id])
                object_id_map[para_id][object] = object_id

            if d[para_id]['questions'][object_id]['question'] == '':
                d[para_id]['questions'][object_id]['question'] = f"the location/state of {object}"

            if step_id == 1: # add init step
                d[para_id]['questions'][object_id]['answers'][step_id - 1] = before

            assert d[para_id]['questions'][object_id]['answers'][step_id - 1] == before or para_id in INVALID_DATA
            d[para_id]['questions'][object_id]['answers'][step_id] = after

    for k, v in d.items():
        # convert dict to list
        v['questions'] = [v['questions'][idx] for idx in range(len(v['questions']))]
        for item in v['questions']:
            item['answers'] = [item['answers'][idx] for idx in range(len(item['answers']))]
        v['steps'] = [v['steps'][x] for x in range(len(v['steps']))]

        for question in v['questions']:
            assert len(v['steps']) == len(question['answers']), k

    for idx in INVALID_DATA:
        if idx in d:
            d.pop(idx)

    print(f"size of: {para_file}: {len(d)}")
    with open(save_file, 'w+') as f:
        json.dump(d, f, indent=2)


if __name__ == "__main__":
    for s in ['train', 'dev', 'test']:
        print(s)
        para_file = f'data/propara/{s}/sentences.tsv'
        answer_file = f'data/propara/{s}/answers.tsv'
        save_file = f'data/propara/{s}.json'
        convert_data(para_file, answer_file, save_file)