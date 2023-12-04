import sys
# sys.path.append("../similarity_grouped_query_attention_pytorch")
sys.path.insert(0, '..')

from config import * 
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention, T5Config, T5Block
from copy import deepcopy
from typing import List
from collections import defaultdict
import torch
import torch.nn.functional as F
import config
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Tokenizer
from datasets import load_dataset
from torch.optim import AdamW
from transformers import get_scheduler
from evaluate import load
import nltk
import numpy as np
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
from t5_SGQA import convert_t5_to_gqa
from reverse_main import preprocess_function,compute_metrics, get_avg

if __name__ == '__main__':
    wandb.login(key=config.WANDB_API_KEY)
    run = wandb.init(project=config.WANDB_PROJECT,config={"model":config.MODEL_NAME,"gqa_list":config.GQA_LIST},entity=config.WANDB_ENTITY,name="MQA")


    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

    t5 = convert_t5_to_gqa(t5,kv_heads=1, similarity_flag=True, inplace=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5)

    data_dir = "data"
    cnn_data_train = load_dataset("cnn_dailymail",data_dir=data_dir,split="train[:100%]")
    cnn_data_test = load_dataset("cnn_dailymail",data_dir=data_dir,split="test[:100%]")
    cnn_data_val = load_dataset("cnn_dailymail",data_dir=data_dir,split="validation[:100%]")

    tokenized_datasets_train = cnn_data_train.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=1000)
    tokenized_datasets_val = cnn_data_val.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=1000)
    tokenized_datasets_test = cnn_data_test.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=1000)

    train_dataloader = DataLoader(tokenized_datasets_train, shuffle=True, batch_size=config.BATCH_SIZE,collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized_datasets_val, batch_size=config.BATCH_SIZE,collate_fn=data_collator)
    test_dataloader = DataLoader(tokenized_datasets_test, batch_size=config.BATCH_SIZE,collate_fn=data_collator)

    num_training_steps = config.NUM_EPOCHS * len(train_dataloader)
    optimizer = AdamW(t5.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    t5.to(device)

    metric = load("rouge")

    progress_bar = tqdm(range(num_training_steps))
    val_rouge_dict = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [], 'gen_len': []}

    for epoch in range(config.NUM_EPOCHS):
        t5.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = t5(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        t5.eval()
        eval_dict_list = []
        for eval_batch in eval_dataloader:
            eval_dict_list.append(compute_metrics(eval_batch,tokenizer,metric))
        
        key_names = eval_dict_list[0].keys()
        average_dict = {k:get_avg(eval_dict_list,k) for k in key_names}
        for k in average_dict.keys():
            val_rouge_dict[k].append(average_dict[k])
    wandb.log({"val_rouge_mqa":val_rouge_dict})
