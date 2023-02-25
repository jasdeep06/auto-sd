import wandb
import json
from utils import create_training_shell_script,get_recent_file_from_directory,run_shell
import uuid
from setup import run_setup
import shutil




def run(train_config):
    wandb.login(key="924764f1e5cac1fa896fada3c8d64b39a0926275")
    # train_config = json.load(open("train_config.json"))
    # run_config = json.load(open("run_config.json"))
    # run_id = run_config['id']
    run_id = uuid.uuid4().hex[0:8]
    train_config['id'] = run_id
    train_config['output_dir'] = train_config['output_dir'].replace('id',run_id)
    train_config['concepts_list'] = train_config['concepts_list'].replace('id',run_id)
    run_setup(run_id)
    with wandb.init(project='generative-ai',job_type='train',config=train_config,name="train-"+run_id) as run:
        train_dataset_name = train_config['train_dataset']
        train_dataset = wandb.use_artifact(f"{train_dataset_name}:latest", type="data")
        regularization_dataset_name = train_config['regularization_dataset']
        regularization_dataset = wandb.use_artifact(f"{regularization_dataset_name}:latest", type="data")
        train_dataset.download(f"/work/{run_id}/subject_images")
        regularization_dataset.download(f"/work/{run_id}/regularization_images")

        



        training_shell_string = create_training_shell_script(train_config,run_id)

        run_shell(training_shell_string)

        model_artifact = wandb.Artifact("trained-model-{}".format(run_id),type="model",description="trained model",metadata=dict(train_config))
        model_artifact.add_dir(get_recent_file_from_directory(f"/work/{run_id}/model_out/*"))
        run.log_artifact(model_artifact)
        shutil.rmtree(f"/work/{run_id}/model_out/")








