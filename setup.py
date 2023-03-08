import json
import wandb
import json
from utils import make_dir,git_clone_in_dir,move_file,run_shell,copy_file
import os


def run_setup(run_id):
    concept_list = json.load(open("concepts_list.json"))

    for i in range(len(concept_list)):
        concept_list[i]['instance_data_dir'] =  concept_list[i]['instance_data_dir'].replace('id',run_id)
        concept_list[i]['class_data_dir'] = concept_list[i]['class_data_dir'].replace('id',run_id)

    # concept_list[0]['instance_data_dir'] =  concept_list[0]['instance_data_dir'].replace('id',run_id)
    # concept_list[0]['class_data_dir'] = concept_list[0]['class_data_dir'].replace('id',run_id)
    json.dump(concept_list,open("concepts_list_temp.json","w"))

    shell_string = ""
    shell_string += make_dir("/work") + " ; "
    shell_string += make_dir("/work/" + run_id) + " ; "

    for i in range(len(concept_list)):
        if not os.path.exists(concept_list[i]["class_data_dir"]):
            print("Making dir",concept_list[i]["class_data_dir"])
            shell_string += make_dir(concept_list[i]["class_data_dir"]) + " ; "
    
    for i in range(len(concept_list)):
        if not os.path.exists(concept_list[i]["instance_data_dir"]):
            print("Making dir",concept_list[i]["instance_data_dir"])
            shell_string += make_dir(concept_list[i]["instance_data_dir"]) + " ; "

    # shell_string += make_dir("/work/" + run_id + "/regularization_images") + " ; "
    # shell_string += make_dir("/work/" + run_id + "/subject_images") + " ; "
    shell_string += make_dir("/work/" + run_id + "/model_out") + " ; "
    shell_string += move_file("concepts_list_temp.json","/work/" + run_id + "/concepts_list.json") + " ; "

    shell_string += git_clone_in_dir("https://github.com/jasdeep06/diffusers.git","/work/" + run_id + "/diffusers") + " ; "
    run_shell(shell_string)









