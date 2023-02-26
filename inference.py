import torch

from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import random
import json
import uuid
import os
import wandb
import shutil





def run(inference_config):
    wandb.login(key="924764f1e5cac1fa896fada3c8d64b39a0926275")
    #inference_config = json.load(open("inference_config.json"))
    inference_id = uuid.uuid4().hex[0:8]
    inference_config['id'] = inference_id
    if not os.path.exists("/work/inference/" + inference_id):
        os.makedirs("/work/inference/" + inference_id)

    with wandb.init(project='generative-ai',job_type='inference',config=inference_config,name="inference-" + inference_id) as run:
        model_name = inference_config['model_name']
        model = wandb.use_artifact(f"{model_name}:latest", type="model")
        model.download(f"/work/inference/{inference_id}/model")

        
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(f"/work/inference/{inference_id}/model", scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16, revision="fp16").to("cuda")

        # Generate the images:
        images = pipe(
                inference_config['prompt'],
                height=inference_config['height'],
                width=inference_config['width'],
                negative_prompt=inference_config['negative_prompt'],
                num_images_per_prompt=inference_config['num_samples'],
                num_inference_steps=inference_config['num_inference_steps'],
                guidance_scale=inference_config['guidance_scale'],
            ).images

        

        # for i in range(len(images)):
        #     wandb.log({"image": wandb.Image(images[i])})
        # inference_table = wandb.Table(columns=["image"], data=[[wandb.Image(img)] for img in images])
        # run.log({"inference_table":inference_table})
        inference_artifact = wandb.Artifact(name=f"output-{inference_id}",type="output")
        for i in range(len(images)):
            inference_artifact.add(wandb.Image(images[i]))
        run.log_artifact(inference_artifact)

    shutil.rmtree(f"/work/inference/{inference_id}/model")


 



# # Change these as you want:
# model_path = "/work/train_1/model_out/1200"
# img_out_folder = "/work/output/"

# # Image related config -- Change if you've used a different keyword:

# # Try our original prompt and make sure it works okay:
# # prompt = "closeup photo of ggd woman in the garden on a bright sunny day"
# prompt = "photo of ggd"
# negative_prompt = ""
# num_samples = 4
# guidance_scale = 8.5
# num_inference_steps = 80
# height = 512
# width = 512

# # Setup the scheduler and pipeline
# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
# pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16, revision="fp16").to("cuda")


# # Generate the images:
# images = pipe(
#         prompt,
#         height=height,
#         width=width,
#         negative_prompt=negative_prompt,
#         num_images_per_prompt=num_samples,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale,
#     ).images

# # Loop on the images and save them:
# for img in images:
#     i = random.randint(0, 200)
#     img.save(f"{img_out_folder}/v2_{i}.png")

    