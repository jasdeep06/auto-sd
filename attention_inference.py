from ane.pipeline_attend_and_excite import AttendAndExcitePipeline
from ane.utils.ptp_utils import AttentionStore,register_attention_control
import torch
from diffusers import DDIMScheduler
from dataclasses import field
import wandb
import random

def run(attention_inference_config):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model_name = attention_inference_config['model_name']
    api = wandb.Api()
    artifact = api.artifact('jasdeep06/generative-ai/{}:latest'.format(model_name))
    artifact.download('model')
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # stable = AttendAndExcitePipeline.from_pretrained(f"/work/inference/{inference_id}/model",scheduler=scheduler).to(device)
    stable = AttendAndExcitePipeline.from_pretrained("model",scheduler=scheduler).to(device)

    token_indices = get_indices_to_alter(stable,attention_inference_config['prompt'],attention_inference_config['tokens_to_alter'])
    num_images = attention_inference_config['num_samples']
    random_seed = attention_inference_config['random_seed']
    for i in range(num_images):
        if not random_seed:
            seed = i
            g = torch.Generator(device=device).manual_seed(seed)
        else:
            seed = random.randint(0,100000)
            g = torch.Generator(device=device).manual_seed(seed)
        controller = AttentionStore()
        register_attention_control(stable,controller)
        image = stable(
            prompt=attention_inference_config['prompt'],
            attention_store = controller,
            indices_to_alter = token_indices,
            attention_res = attention_inference_config['attention_res'],
            guidance_scale = attention_inference_config['guidance_scale'],
            generator = g,
            num_inference_steps = attention_inference_config['num_inference_steps'],
            max_iter_to_alter = attention_inference_config['max_iter_to_alter'],
            run_standard_sd = False,
            thresholds = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor = attention_inference_config['scale_factor'],
            scale_range = (1.0, 0.5),
            smooth_attentions = attention_inference_config['smooth_attentions'],
            sigma = attention_inference_config['sigma'],
            kernel_size = attention_inference_config['kernel_size'],
            sd_2_1 = True,
            negative_prompt = attention_inference_config['negative_prompt']
        ).images[0]

        image.save(f"image_{seed}.png")


def get_indices_to_alter(stable,prompt,tokens_to_alter):
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    word_to_token_idx = {v: k for k, v in token_idx_to_word.items()}
    indices_to_alter = [word_to_token_idx[t] for t in tokens_to_alter]
    return indices_to_alter



