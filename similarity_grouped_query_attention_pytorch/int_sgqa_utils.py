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
# from t5_WGQA import convert_t5_to_wgqa
from t5_WGQA_final import convert_t5_to_wgqa
import torch.nn as nn
import torch.distributed as dist
import os
import shutil
# from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, T5Tokenizer
from t5_SGQA import convert_t5_to_gqa


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

def validation_loop(t5,tokenizer,metric,eval_dataloader,step,device):
    epoch_eval_loss = []
    eval_dict_list = []
    print(f"Started evaluation for step {step}")
    for eval_batch in eval_dataloader:
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        eval_outputs = t5(**eval_batch)
        eval_loss = eval_outputs.loss
        epoch_eval_loss.append(eval_loss.item())
        eval_batch_pred_tensors = t5.generate(eval_batch['input_ids'],max_length=config.MAX_TARGET_LENGTH)
        val_rouge_step_metric = compute_metrics(eval_batch_pred_tensors.cpu(), eval_batch['labels'].cpu(), tokenizer, metric)
        eval_dict_list.append(val_rouge_step_metric)
    mean_eval_loss = sum(epoch_eval_loss)/len(epoch_eval_loss)
    return mean_eval_loss,eval_dict_list

def testing_loop(t5,tokenizer,metric,test_dataloader,device):
    test_dict_list = []
    for test_batch in test_dataloader:
        test_batch = {k: v.to(device) for k, v in test_batch.items()}
        test_batch_pred_tensors = t5.generate(test_batch['input_ids'],max_length=config.MAX_TARGET_LENGTH)
        test_dict_list.append(compute_metrics(test_batch_pred_tensors.cpu(),test_batch['labels'].cpu(),tokenizer,metric))
    
    return test_dict_list

def repeat_kv_heads(t5_gqa,d_model,kv_heads,n_heads):
    for layer in t5_gqa.decoder.block:
        # print(layer.layer[0].SelfAttention.q)
        curr_self_attention_layer = layer.layer[0].SelfAttention
        k_weight_data = curr_self_attention_layer.k.weight.data
        k_weight_data = k_weight_data.view(kv_heads,d_model//n_heads,d_model)
        k_weight_data = torch.repeat_interleave(k_weight_data,2,dim=1).view(-1,d_model)
        
        v_weight_data = curr_self_attention_layer.v.weight.data
        v_weight_data = v_weight_data.view(kv_heads,d_model//n_heads,d_model)
        v_weight_data = torch.repeat_interleave(v_weight_data,2,dim=1).view(-1,d_model)
        
        curr_self_attention_layer.k = nn.Linear(in_features=512,out_features=512,bias=False)
        curr_self_attention_layer.v = nn.Linear(in_features=512,out_features=512,bias=False)
        
        curr_self_attention_layer.k.weight.data = k_weight_data
        curr_self_attention_layer.v.weight.data = v_weight_data

        curr_cross_attention_layer = layer.layer[1].EncDecAttention
        k_weight_data = curr_cross_attention_layer.k.weight.data
        k_weight_data = k_weight_data.view(kv_heads,d_model//n_heads,d_model)
        k_weight_data = torch.repeat_interleave(k_weight_data,2,dim=1).view(-1,d_model)
        
        v_weight_data = curr_cross_attention_layer.v.weight.data
        v_weight_data = v_weight_data.view(kv_heads,d_model//n_heads,d_model)
        v_weight_data = torch.repeat_interleave(v_weight_data,2,dim=1).view(-1,d_model)
        
        curr_cross_attention_layer.k = nn.Linear(in_features=512,out_features=512,bias=False)
        curr_cross_attention_layer.v = nn.Linear(in_features=512,out_features=512,bias=False)
        
        curr_cross_attention_layer.k.weight.data = k_weight_data
        curr_cross_attention_layer.v.weight.data = v_weight_data
    return t5_gqa

def train(rank,world_size,kv_heads:int,logging_name:str,run,model_name:str=config.MODEL_NAME,similarity_flag:bool=False,short_or_long:str="short"):
    logging_name = short_or_long.upper() + "_" + logging_name
    assert short_or_long in ["short","long"],  "Please only provide short or long"

    if short_or_long == "short": time_interval = config.SHORT_SIMILARITY_INTERVAL
    elif short_or_long == "long": time_interval = config.LONG_SIMILARITY_INTERVAL

    dir = logging_name.upper()
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    device = torch.device("cuda", rank)
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        model_name
    )
    #Initially have similarity as False
    device = torch.device("cuda")
    t5 = convert_t5_to_gqa(t5,kv_heads=kv_heads,similarity_flag=False,inplace=True)
    t5.to(device)
    # t5 = torch.nn.parallel.DistributedDataParallel(t5, device_ids=[rank],find_unused_parameters=True)

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
    
    cnn_data_train = load_dataset("cnn_dailymail",data_dir=data_dir,split=f"train[:{config.PERCENT_DATA}%]")
    cnn_data_test = load_dataset("cnn_dailymail",data_dir=data_dir,split=f"test[:{config.PERCENT_DATA}%]")
    cnn_data_val = load_dataset("cnn_dailymail",data_dir=data_dir,split=f"validation[:{config.PERCENT_DATA}%]")
    
    tokenized_datasets_train = cnn_data_train.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=config.TOKENIZE_BATCH_SIZE)
    tokenized_datasets_val = cnn_data_val.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=config.TOKENIZE_BATCH_SIZE)
    tokenized_datasets_test = cnn_data_test.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=config.TOKENIZE_BATCH_SIZE)

    
    # train_sampler = DistributedSampler(tokenized_datasets_train)
    train_dataloader = DataLoader(tokenized_datasets_train,shuffle=True, batch_size=config.BATCH_SIZE,collate_fn=data_collator)
    
    # eval_sampler = DistributedSampler(tokenized_datasets_val, shuffle=False)
    eval_dataloader = DataLoader(tokenized_datasets_val, batch_size=config.VAL_BATCH_SIZE,collate_fn=data_collator)
    
    # test_sampler = DistributedSampler(tokenized_datasets_test, shuffle=False)
    test_dataloader = DataLoader(tokenized_datasets_test, batch_size=config.VAL_BATCH_SIZE,collate_fn=data_collator)

    num_training_steps = config.NUM_EPOCHS * len(train_dataloader)
    optimizer = AdamW(t5.parameters(), lr=config.LEARNING_RATE )
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    metric = load("rouge")

    progress_bar = tqdm(range(num_training_steps))
    val_rouge_dict = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [], 'gen_len': []}

    train_loss_list = []
    val_loss_list = []
    steps = 0
    for epoch in range(config.NUM_EPOCHS):
        t5.train()
        epoch_train_loss = []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = t5(**batch)
            loss = outputs.loss
            loss.backward()
            epoch_train_loss.append(loss.item())
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            steps+=1
            if steps%time_interval==0:
                torch.save(t5.state_dict(),f"{dir}/{logging_name.lower()}_t5_finetuned_step_{epoch}_before.pth")
                t5 = repeat_kv_heads(t5,d_model=512,kv_heads=kv_heads,n_heads=8)
                t5 = convert_t5_to_gqa(t5,kv_heads=kv_heads,similarity_flag=similarity_flag,inplace=True)
                # t5.to(device)
                # t5 = torch.nn.parallel.DistributedDataParallel(t5, device_ids=[rank],find_unused_parameters=True)
                torch.save(t5.state_dict(),f"{dir}/{logging_name.lower()}_t5_finetuned_step_{epoch}_after.pth")

                
        mean_train_loss = sum(epoch_train_loss)/len(epoch_train_loss)
        train_loss_list.append(mean_train_loss)

        t5.eval()
        if rank == 0:
            mean_eval_loss,eval_dict_list = validation_loop(t5,tokenizer,metric,eval_dataloader,steps,device)
            key_names = eval_dict_list[0].keys()
            average_dict = {k:get_avg(eval_dict_list,k) for k in key_names}
            for k in average_dict.keys():
                val_rouge_dict[k].append(average_dict[k])
            print(f'Epoch: {epoch} val rogue {val_rouge_dict}')
            run.log({f"{logging_name.lower()}_val_epoch_{short_or_long}"+k:v[0] for k,v in val_rouge_dict.items()})
        # print(rank)
        if rank==0:
            print(f"Started testing for step {steps}")
            test_dict_list = testing_loop(t5,tokenizer,metric,test_dataloader,device)
            key_names = test_dict_list[0].keys()
            test_rouge_dict = {k:get_avg(test_dict_list,k) for k in key_names}
            print(f'Epoch: {epoch} test rogue {test_rouge_dict}')
            run.log({f"{logging_name.lower()}_test_epoch_{short_or_long}"+k:v for k,v in test_rouge_dict.items()})
        
        print(f'Train and val loss after {epoch} epoch:{mean_train_loss}, val:{mean_eval_loss}')
        run.log({
            "Train Loss":mean_train_loss,
            "Val Loss":mean_eval_loss
        })
            
        if rank==0:
            t5.eval()
            torch.save(t5.state_dict(),f"{dir}/{logging_name.lower()}_t5_finetuned_epoch_{epoch}.pth")

    return val_rouge_dict,test_rouge_dict



