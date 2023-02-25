import json
from inference import run

def combinations(lists):
    # Base case: if the input list is empty, return an empty list
    if not lists:
        return []
    # Recursive step: generate combinations recursively
    if len(lists) == 1:
        return [(item,) for item in lists[0]]
    else:
        result = []
        for item in lists[0]:
            print(item)
            for subcombination in combinations(lists[1:]):
                result.append((item,) + subcombination)
        return result


inference_run_config = json.load(open("inference_runs_config.json"))
param_keys = list(inference_run_config.keys())
param_values = list(inference_run_config.values())
param_combinations = combinations(param_values)

inference_params = [dict(zip(param_keys, param_combination)) for param_combination in param_combinations]
inference_config = json.load(open("inference_config.json"))

for inference_param in inference_params:
    for key,value in inference_param.items():
        inference_config[key] = value
    print("running ",inference_config)
    run(inference_config)

