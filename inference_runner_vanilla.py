import wandb
import json
from inference import run_without_wandb_run


wandb.login(key="924764f1e5cac1fa896fada3c8d64b39a0926275")
inference_config = json.load(open("inference_config.json"))
print("running ",inference_config)
run_without_wandb_run(inference_config)

