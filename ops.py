import wandb

def upload_artifact(project,artifact_name,artifact_type,artifact_path,description):
    with wandb.init(project=project,job_type='upload') as run:
        raw_data = wandb.Artifact(name=artifact_name,type=artifact_type,description=description)
        raw_data.add_dir(artifact_path)
        run.log_artifact(raw_data)

def download_artifact(project,artifact_name,artifact_root):
    with wandb.init(project=project,job_type='download') as run:
        artifact = run.use_artifact(name=artifact_name, type="dataset")
        artifact_dir = artifact.download(artifact_root)
        print(artifact_dir)

# upload_artifact("generative-ai",
# "regularization-sikh","data",
# "X:\\python_projects\\vinglabs\\generative-ai\\stable-diffusion\\data-prep\\sikh_images",
# "regularization images of sikh man")