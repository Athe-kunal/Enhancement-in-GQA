import sys
# sys.path.append("../similarity_grouped_query_attention_pytorch")
sys.path.insert(0, '..')
import wandb
import argparse
from config import *
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Directory of files to upload to wandb')
    parser.add_argument('-dname','--dir_name',type=str,help='directory name')
    args = parser.parse_args()
    dir_name = args.dir_name.upper()
    assert dir_name in os.listdir(".")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT,config={"type":"model_files"},entity=WANDB_ENTITY,group=dir_name+"_MODEL")

    # for model_file in os.listdir(dir_name):
        
    #     artifact = wandb.Artifact(name=model_file,type="model")
    #     artifact.add_file(local_path=os.path.join(dir_name,model_file))
    #     run.log_artifact(artifact)
    artifact = wandb.Artifact(name=dir_name,type="model")
    artifact.add_dir(local_path=dir_name)
    run.log_artifact(artifact)
