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
from transformers import AutoTokenizer
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
import torch.nn as nn
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler

wandb.login(key=config.WANDB_API_KEY)
run = wandb.init(project=config.WANDB_PROJECT,config={"model":config.MODEL_NAME,"gqa_list":config.GQA_LIST},entity=config.WANDB_ENTITY)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def compute_metrics(predictions,labels,tokenizer,metric):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Note that other metrics may not have a `use_aggregator` parameter
    # and thus will return a list, computing a metric for each sentence.
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

def get_avg(eval_dict_list,key_name:str):
    return sum(d[key_name] for d in eval_dict_list) / len(eval_dict_list)


def train(rank,world_size,model_name:str=config.MODEL_NAME):
    device = torch.device("cuda", rank)
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        model_name
    )
    t5 = convert_t5_to_gqa(t5,kv_heads=4)
    t5.to(rank)
    t5 = torch.nn.parallel.DistributedDataParallel(t5, device_ids=[rank])

    tokenizer =  AutoTokenizer.from_pretrained(model_name,legacy=False)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5)

    def preprocess_function(examples,max_input_length:int=config.MAX_INPUT_LENGTH,max_target_length:int=config.MAX_TARGET_LENGTH):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_dir = "data"
    
    cnn_data_train = load_dataset("cnn_dailymail",data_dir=data_dir,split="train[:100%]")
    cnn_data_test = load_dataset("cnn_dailymail",data_dir=data_dir,split="test[:100%]")
    cnn_data_val = load_dataset("cnn_dailymail",data_dir=data_dir,split="validation[:100%]")
    
    tokenized_datasets_train = cnn_data_train.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=1000)
    tokenized_datasets_val = cnn_data_val.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=1000)
    tokenized_datasets_test = cnn_data_test.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=1000)

    
    train_sampler = DistributedSampler(tokenized_datasets_train)
    train_dataloader = DataLoader(tokenized_datasets_train, batch_size=config.BATCH_SIZE, sampler=train_sampler, collate_fn=data_collator)
    
    eval_sampler = DistributedSampler(tokenized_datasets_val, shuffle=False)
    eval_dataloader = DataLoader(tokenized_datasets_val, batch_size=config.BATCH_SIZE,collate_fn=data_collator,sampler=eval_sampler)
    
    test_sampler = DistributedSampler(tokenized_datasets_test, shuffle=False)
    test_dataloader = DataLoader(tokenized_datasets_test, batch_size=config.BATCH_SIZE,collate_fn=data_collator,sampler=test_sampler)

    num_training_steps = config.NUM_EPOCHS * len(train_dataloader)
    optimizer = AdamW(t5.parameters(), lr=5e-5)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # t5.to(device)

    # t5 = nn.DataParallel(t5)
    # t5.to('cuda')
    # device = 'cuda'

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
            # print(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            # break
        t5.eval()
        if rank == 0:
            eval_dict_list = []
            print(f"Started evaluation for epoch {epoch}")
            for eval_batch in eval_dataloader:
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                eval_batch_pred_tensors = t5.module.generate(eval_batch['input_ids'])
                eval_dict_list.append(compute_metrics(eval_batch_pred_tensors.cpu(), eval_batch['labels'].cpu(), tokenizer, metric))
            torch.save(t5.module.state_dict(), "t5_finetuned.pth")
        # eval_dict_list = []
        # print(f"Started evaluation for epoch {epoch}")
        # for eval_batch in tqdm(eval_dataloader):
        #     eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        #     eval_batch_pred_tensors = t5.generate(eval_batch['input_ids'])
        #     eval_dict_list.append(compute_metrics(eval_batch_pred_tensors.cpu(),eval_batch['labels'].cpu(),tokenizer,metric))
        
            key_names = eval_dict_list[0].keys()
            average_dict = {k:get_avg(eval_dict_list,k) for k in key_names}
            for k in average_dict.keys():
                val_rouge_dict[k].append(average_dict[k])
            print(f'Epoch==.{epoch} val rogue {val_rouge_dict}')
            wandb.log({f"val_rouge_{epoch}":val_rouge_dict})
        # break
    wandb.log({"val_rouge":val_rouge_dict})
    if rank==0:
        test_dict_list = []
        for test_batch in test_dataloader:
            test_batch = {k: v.to(device) for k, v in test_batch.items()}
            test_batch_pred_tensors = t5.module.generate(test_batch['input_ids'])
            test_dict_list.append(compute_metrics(test_batch_pred_tensors.cpu(),test_batch['labels'].cpu(),tokenizer,metric))
        
        key_names = test_dict_list[0].keys()
        test_rouge_dict = {k:get_avg(test_dict_list,k) for k in key_names}
        wandb.log({"test_rouge":test_rouge_dict})
        # Save only on the master process
        torch.save(t5.module.state_dict(), "t5_finetuned.pth")

        return val_rouge_dict,test_rouge_dict

def main(rank, world_size):
    setup(rank, world_size)
    val_rouge_dict, test_rouge_dict = train(rank, world_size)
    print(f'validation rogue dict:{val_rouge_dict}')
    print(f'Test rogue dict:{test_rouge_dict}')
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    


