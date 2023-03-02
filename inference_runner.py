import json
from inference import run
from utils import get_wandb_artifacts,filter_model_based_on_metadata_range
import wandb

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
            for subcombination in combinations(lists[1:]):
                result.append((item,) + subcombination)
        return result

wandb.login(key="924764f1e5cac1fa896fada3c8d64b39a0926275")
inference_run_config = json.load(open("inference_runs_config.json"))
inference_run_config['model_name'] = filter_model_based_on_metadata_range("learning_rate",0.0000000001,0.0001)
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

