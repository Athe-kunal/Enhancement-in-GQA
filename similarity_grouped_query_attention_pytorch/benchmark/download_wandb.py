import sys
# sys.path.append("../similarity_grouped_query_attention_pytorch")
sys.path.insert(0, '..')
import wandb
import argparse
from config import *
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Directory of files to download from wandb')
    parser.add_argument('-dname','--dir_name',type=str,help='directory name')
    args = parser.parse_args()
    dir_name = args.dir_name.upper()
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=WANDB_PROJECT,config={"type":"model_files"},entity=WANDB_ENTITY)

    artifact = run.use_artifact(f'athe_kunal/similarity_gqa/{dir_name}:v0', type='model')
    artifact_dir = artifact.download(root=dir_name)
    run.finish()