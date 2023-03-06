import json
import subprocess
import os
import wandb
import time
from glob import glob
import cv2
import numpy as np
import time

def make_dir(dirname):
    return "mkdir {}".format(dirname)

def git_clone_in_dir(repoaddr,dirname):
    return "git clone {} {}".format(repoaddr,dirname)

def change_dir(dirname):
    return "cd {}".format(dirname)

def move_file(src,dest):
    return "mv {} {}".format(src,dest)

def create_training_shell_script(train_config,run_id):
    """
    accelerate launch train_dreambooth.py \
 --gradient_accumulation_steps=1 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
 --pretrained_vae_name_or_path "stabilityai/sd-vae-ft-mse" --output_dir=/work/train_1/model_out/ --with_prior_preservation \
 --prior_loss_weight=1.0 --resolution=512 --train_batch_size=1 --learning_rate=2e-6 \
 --lr_scheduler="constant" --lr_warmup_steps=0 --num_class_images=200 --max_train_steps=1200 \
 --concepts_list="/work/train_1/concepts_list.json" --train_text_encoder --revision="fp16" --mixed_precision="fp16"
    """

    gradient_accumulation_steps = train_config['gradient_accumulation_steps']
    pretrained_model_name_or_path = train_config['pretrained_model_name_or_path']
    pretrained_vae_name_or_path = train_config['pretrained_vae_name_or_path']
    output_dir = train_config['output_dir']
    with_prior_preservation = train_config['with_prior_preservation']
    prior_loss_weight = train_config['prior_loss_weight']
    resolution = train_config['resolution']
    train_batch_size = train_config['train_batch_size']
    learning_rate = train_config['learning_rate']
    lr_scheduler = train_config['lr_scheduler']
    lr_warmup_steps = train_config['lr_warmup_steps']
    num_class_images = train_config['num_class_images']
    max_train_steps = train_config['max_train_steps']
    concepts_list = train_config['concepts_list']
    train_text_encoder = train_config['train_text_encoder']
    revision = train_config['revision']
    mixed_precision = train_config['mixed_precision']
    new_tokens = train_config['new_tokens']

    shell_string = ""
    shell_string += "accelerate launch /work/{}/diffusers/examples/dreambooth/train_dreambooth.py ".format(run_id)
    shell_string += "--gradient_accumulation_steps={} ".format(gradient_accumulation_steps)
    shell_string += "--pretrained_model_name_or_path={} ".format(pretrained_model_name_or_path)
    shell_string += "--pretrained_vae_name_or_path={} ".format(pretrained_vae_name_or_path)
    shell_string += "--output_dir={} ".format(output_dir)
    if with_prior_preservation:
        shell_string += "--with_prior_preservation "
    shell_string += "--prior_loss_weight={} ".format(prior_loss_weight)
    shell_string += "--resolution={} ".format(resolution)
    shell_string += "--train_batch_size={} ".format(train_batch_size)
    shell_string += "--learning_rate={} ".format(learning_rate)
    shell_string += "--lr_scheduler={} ".format(lr_scheduler)
    shell_string += "--lr_warmup_steps={} ".format(lr_warmup_steps)
    shell_string += "--num_class_images={} ".format(num_class_images)
    shell_string += "--max_train_steps={} ".format(max_train_steps)
    shell_string += "--concepts_list={} ".format(concepts_list)
    if train_text_encoder:
        shell_string += "--train_text_encoder "
    shell_string += "--revision={} ".format(revision)
    shell_string += "--mixed_precision={} ".format(mixed_precision)
    if bool(new_tokens):
        shell_string += "--new_tokens={} ".format(" ".join(new_tokens))

    return shell_string

def run_shell(shell_string):
    p2 = subprocess.Popen(shell_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p2.stdout:
        print(line.decode(), end='')

def get_recent_file_from_directory(dir_name):
    list_of_files = glob(dir_name)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def copy_file(src,dest):
    return "cp {} {}".format(src,dest)

def get_wandb_artifacts(artifact_type):
    api = wandb.Api()
    # wandb.init()
    # return [coll.name for coll in api.artifact_type(artifact_type,project='generative-ai').collections()]
    #return [artifact.name for artifact in wandb.artifacts() if artifact.type == "model"]
    artifact_names = []
    for run in api.runs('jasdeep06/generative-ai',filters={"created_at":{"$gt":"2023-02-24T00:00:00","$lt":"2023-02-26T00:00:00}}"}}):
        # artifacts = api.artifacts(run.path)
        # for artifact in artifacts:
        #     if artifact.type == "model":
        #         artifact_names.append(artifact.name)
        for artifact in run.logged_artifacts():
            if artifact.type == "model":
                artifact_names.append(artifact.name.split(":")[0])
    return artifact_names


def get_runs_using_artifact(artifact_name):
    filtered_runs = []
    api = wandb.Api()
    runs = api.runs("jasdeep06/generative-ai")
    for run in runs:
        # if 'trained-model-500d47b9' in run.logged_artifacts():
        for art in run.used_artifacts():
            if art.name.split(":")[0] == artifact_name:
                filtered_runs.append(run.name)
    return filtered_runs


def get_metadata_from_artifact(artifact_name):
    api = wandb.Api()
    runs = api.runs("jasdeep06/generative-ai")
    for run in runs:
        for art in run.logged_artifacts():
            if art.name.split(":")[0] == artifact_name:
                return art.metadata

def download_inference_run_outputs():
    api = wandb.Api(timeout=100)
    runs = api.runs("jasdeep06/generative-ai",filters={"created_at":{"$gt":"2023-03-01T16:30:00","$lt":"2023-03-03T06:00:00"}})
    print(len(runs))
    for i,run in enumerate(runs):
        print("Downloading run {} of {}".format(i+1,len(runs)))
        print(run.name)
        for artifact in run.logged_artifacts():
            if artifact.type == "output":
                dest = os.path.join("outputs",run.name)
                os.mkdir(dest)
                artifact.download(dest)

def filter_model_based_on_metadata_range(key,minval,maxval):
    model_names = []
    api = wandb.Api()
    runs = api.runs("jasdeep06/generative-ai")
    for run in runs:
        for art in run.logged_artifacts():
            if art.type == "model":
                if art.metadata[key] >= minval and art.metadata[key] <= maxval:
                    model_names.append(art.name.split(":")[0])
    return model_names

def get_runs_and_metadata_using_artifact_optimised(artifact_name):
    runs = []
    api = wandb.Api()
# runs = api.runs("jasdeep06/generative-ai",filters={"used_artifacts":"jasdeep06/generative-ai/trained-model-500d47b9:latest"})
    art = api.artifact("jasdeep06/generative-ai/{}:latest".format(artifact_name))
    for run in art.used_by():
        runs.append(run.name)
    return runs,art.metadata

# print(filter_model_based_on_metadata_range("learning_rate",0.0000000001,0.0001))
# print(runs)
# for run in runs:
#     print(run)
# api = wandb.Api()
# artifact = api.artifact('jasdeep06/generative-ai/output-trained-model-500d47b9-3:latest')
# artifact.download()
# print(get_metadata_from_artifact('trained-model-500d47b9'))
# print(get_runs_using_artifact('trained-model-500d47b9'))

# download_inference_run_outputs()

# api = wandb.Api()
# artifact = api.artifact(name='jasdeep06/generative-ai/trained-model-b29dd1bf:latest',type='model')
# for run_paths in artifact.used_by():
#     print(run_paths)
# t1 = time.time()
# runs,meta = get_runs_and_metadata_using_artifact_optimised('trained-model-500d47b9')
# print(runs)
# print(meta)
# print(time.time()-t1)
# t2 = time.time()
# print(get_metadata_from_artifact('trained-model-500d47b9'))
# print(time.time()-t2)


def make_grid(output_dir):
    #models = filter_model_based_on_metadata_range("learning_rate",0.0000000001,0.0001)
    #print(models)
    models = ['trained-model-500d47b9', 'trained-model-8c8cc399', 'trained-model-652e79b8', 'trained-model-b29dd1bf', 'trained-model-1']
    inference_runs_config = json.load(open("inference_runs_config.json"))
    num_inference_steps = inference_runs_config["num_inference_steps"]
    guidance_scale = inference_runs_config["guidance_scale"]
    grid = {}
    output_grid = {}
    for model in models:
        model_id = model.split("-")[-1]
        grid[model_id] = []
        model_outputs = glob("outputs/inference-{}-*".format(model_id))
        for model_output in model_outputs:
            metadata = json.load(open(os.path.join(model_output,"metadata.json")))
            data = [metadata['inference_config']['num_inference_steps'],metadata['inference_config']['guidance_scale']]
            image_paths = glob(os.path.join(model_output,"*.png"))
            for image_path in image_paths:
                data.append(cv2.imread(image_path))
            
            grid[model_id].append(data)
    for key,value in grid.items():
        sorted_grid = sorted(value, key=lambda x: (x[0], x[1]))
        sorted_images = [datum[2:] for datum in sorted_grid]
        num_images = len(sorted_images[0])
        horizontal_images = []
        for i in range(num_images):
            horizontal_images.append([])
        for sorted_image in sorted_images:
            for i in range(num_images):
                horizontal_images[i].append(sorted_image[i])
        # for i in range(0,len(sorted_images),len(num_inference_steps)):
        #     horizontal_images.append(np.hstack(sorted_images[i:i+len(num_inference_steps)]))
        # grid[key] = np.vstack(horizontal_images)
        for i,horizontal_image in enumerate(horizontal_images):
            horizontal_images[i] = [np.hstack(horizontal_image[j:j+len(num_inference_steps)]) for j in range(0,len(horizontal_image),len(num_inference_steps))]
            if key in output_grid.keys():
                output_grid[key].append(np.vstack(horizontal_images[i]))
            else:
                output_grid[key] = [np.vstack(horizontal_images[i])]
    if not os.path.exists("formatted_outputs"):
        os.mkdir("formatted_outputs")

    for key,value in output_grid.items():
        if not os.path.exists(os.path.join("formatted_outputs",key)):
            os.mkdir(os.path.join("formatted_outputs",key))
        for i,indi_val in enumerate(value):
            cv2.imwrite(os.path.join("formatted_outputs",key,"{}.jpg".format(i)),indi_val)

            
    
    # for key,value in grid.items():
    #     cv2.imwrite("grid_{}.png".format(key),value)

    

                
                

# make_grid("outputs")
