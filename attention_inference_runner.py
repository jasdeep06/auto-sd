import wandb
import json
from attention_inference import run


wandb.login(key="924764f1e5cac1fa896fada3c8d64b39a0926275")
attention_inference_config = json.load(open("attention_inference_config.json"))
print("running ",attention_inference_config)
run(attention_inference_config)