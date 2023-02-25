import json
import subprocess
import os
import glob

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

    return shell_string

def run_shell(shell_string):
    p2 = subprocess.Popen(shell_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p2.stdout:
        print(line.decode(), end='')

def get_recent_file_from_directory(dir_name):
    list_of_files = glob.glob(dir_name)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def copy_file(src,dest):
    return "cp {} {}".format(src,dest)
