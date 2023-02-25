import json
import numpy as np
from train import run

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


def sample(config):
    sample = {}
    for key,value in config.items():
        if key not in sample.keys():
            sample[key] = []
        frequency = value[-1]
        if key == 'learning_rate':
            for i in range(frequency):
                log_sampling = np.random.uniform(np.log10(value[0]),np.log10(value[1]))
                sample[key].append(10**log_sampling)
        elif key == 'max_train_steps':
            for i in range(frequency):
                sample[key].append(int(np.random.uniform(value[0],value[1])))
    return sample
            




train_runs_config = json.load(open("train_runs_config.json"))
sampled_config = sample(train_runs_config)
param_keys = list(sampled_config.keys())
param_values = list(sampled_config.values())
param_combinations = combinations(param_values)

train_params = [dict(zip(param_keys, param_combination)) for param_combination in param_combinations]

for train_param in train_params:
    train_config = json.load(open("train_config.json"))
    for key,value in train_param.items():
        train_config[key] = value
    print("running ",train_config)
    run(train_config)


