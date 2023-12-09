import sys
sys.path.insert(0, '..')
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Attention, T5Config, T5Block
from copy import deepcopy
from typing import List
from collections import defaultdict
import torch
import torch.nn.functional as F
import checkpoint_config as chk_config
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
from transformers import AutoTokenizer, T5Tokenizer
import json

def load_model(checkpoint_path,model_name):
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("t5-small")

    if model_name == 'GQA':
        t5_finetuned = convert_t5_to_gqa(t5,kv_heads=4,similarity_flag=False)
    elif model_name == 'WGQA':
        t5_finetuned = convert_t5_to_wgqa(t5,kv_heads=4,weight_flag=True,if_random=False)
    elif model_name == 'SIMGQA':
        t5_finetuned = convert_t5_to_gqa(t5,kv_heads=4,similarity_flag=True)
    elif model_name == 'RAND_WGQA':
        t5_finetuned = convert_t5_to_wgqa(t5,kv_heads=4,weight_flag=True,if_random=True)
    elif model_name == 'MQA':
        t5_finetuned = convert_t5_to_gqa(t5,kv_heads=1,similarity_flag=False)
    else:
        print(f'{model_name} not found!')
    
    del t5
    # Load state dict
    t5_finetuned.load_state_dict(torch.load(checkpoint_path))
    
    return t5_finetuned

def get_avg(eval_dict_list,key_name:str):
    return sum(d[key_name] for d in eval_dict_list) / len(eval_dict_list)

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

def validation_loop(t5,tokenizer,metric,eval_dataloader,device):
    epoch_eval_loss = []
    eval_dict_list = []
    for eval_batch in tqdm(eval_dataloader):
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        eval_outputs = t5(**eval_batch)
        eval_loss = eval_outputs.loss
        epoch_eval_loss.append(eval_loss.item())
        del eval_outputs
        eval_batch_pred_tensors = t5.generate(eval_batch['input_ids'],max_length=chk_config.MAX_TARGET_LENGTH)
        val_rouge_step_metric = compute_metrics(eval_batch_pred_tensors.cpu(), eval_batch['labels'].cpu(), tokenizer, metric)
        eval_dict_list.append(val_rouge_step_metric)
    mean_eval_loss = sum(epoch_eval_loss)/len(epoch_eval_loss)
    return mean_eval_loss,eval_dict_list

def testing_loop(t5,tokenizer,metric,test_dataloader,device):
    test_dict_list = []
    for test_batch in tqdm(test_dataloader):
        test_batch = {k: v.to(device) for k, v in test_batch.items()}
        test_batch_pred_tensors = t5.generate(test_batch['input_ids'],max_length=chk_config.MAX_TARGET_LENGTH)
        test_dict_list.append(compute_metrics(test_batch_pred_tensors.cpu(),test_batch['labels'].cpu(),tokenizer,metric))
    
    return test_dict_list

def checkpoint_results(run,models_info):
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer =  AutoTokenizer.from_pretrained(chk_config.MODEL_NAME,legacy=False)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=t5)
    
    device = torch.device("cuda")
    
    del t5

    def preprocess_function(examples,max_input_length:int=chk_config.MAX_INPUT_LENGTH,max_target_length:int=chk_config.MAX_TARGET_LENGTH):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["article"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,padding=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["highlights"], max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_dir = "data"

    cnn_data_test = load_dataset("cnn_dailymail",data_dir=data_dir,split=f"test[:{chk_config.PERCENT_DATA}%]")
    cnn_data_val = load_dataset("cnn_dailymail",data_dir=data_dir,split=f"validation[:{chk_config.PERCENT_DATA}%]")

    tokenized_datasets_val = cnn_data_val.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=chk_config.TOKENIZE_BATCH_SIZE)
    tokenized_datasets_test = cnn_data_test.map(preprocess_function, batched=True,remove_columns=['article','highlights','id'],batch_size=chk_config.TOKENIZE_BATCH_SIZE)
    
    eval_dataloader = DataLoader(tokenized_datasets_val, batch_size=chk_config.VAL_BATCH_SIZE,collate_fn=data_collator)
    test_dataloader = DataLoader(tokenized_datasets_test, batch_size=chk_config.VAL_BATCH_SIZE,collate_fn=data_collator)

    metric = load("rouge")


    # models_info =['GQA','MQA','SIMGQA','WGQA','RANDWGQA']
    val_results = {}
    test_results = {}
    #iterate through each model
    checkpoints = ['_t5_finetuned_steps_2000.pth',
                   '_t5_finetuned_steps_4000.pth',
                    '_t5_finetuned_steps_6000.pth',
                    '_t5_finetuned_steps_8000.pth'
                    '_t5_finetuned_epoch_0.pth',
                    '_t5_finetuned_steps_10000.pth',
                    '_t5_finetuned_steps_12000.pth',
                    '_t5_finetuned_steps_14000.pth',
                    '_t5_finetuned_steps_16000.pth',
                    '_t5_finetuned_epoch_1.pth',
                    '_t5_finetuned_steps_18000.pth',
                    '_t5_finetuned_steps_20000.pth',
                    '_t5_finetuned_steps_22000.pth',
                    '_t5_finetuned_steps_24000.pth',
                    '_t5_finetuned_steps_26000.pth',
                    '_t5_finetuned_epoch_2.pth',]

    for model_name in [models_info]:
        model_val_results = {}
        model_test_results = {}
        if model_name in os.listdir(os.getcwd()):
            curr_folder = os.path.join(os.getcwd(),model_name) #get current folder

            val_results[model_name]={}
            test_results[model_name] = {}

            model_val_results[model_name] = {}
            model_test_results[model_name] = {}

            for chk_point in checkpoints: #iterate through each check point
                file_name = model_name.lower()+chk_point
                
                file_path = os.path.join(curr_folder,file_name)
                t5_finetuned = load_model(file_path,model_name)
                
                t5_finetuned.to(device)
                
                _,_,_,step_type,step_size = chk_point.split('.')[0].split('_') #['_', 't5', 'finetuned', 'steps', '100000']

                t5_finetuned.eval()
                print(f"Started validation for {file_name}")
                mean_eval_loss,eval_dict_list = validation_loop(t5_finetuned,tokenizer,metric,eval_dataloader,device)
                key_names = eval_dict_list[0].keys()
                val_rouge_dict = {k:get_avg(eval_dict_list,k) for k in key_names}
                val_rouge_dict['loss'] = mean_eval_loss
                
                if step_type=='epoch':
                    step_size = str((int(step_size)+1)*8973)
                
                print(f'Model name:{model_name} file_name: {file_name} val rogue {val_rouge_dict}')
                
                for metric_name, metric_value in val_rouge_dict.items():
                    run.log({"val_" + model_name + '_'+metric_name: metric_value})

                print(f"Started testing for {file_name}")
                test_dict_list = testing_loop(t5_finetuned,tokenizer,metric,test_dataloader,device)
                key_names = test_dict_list[0].keys()
                test_rouge_dict = {k:get_avg(test_dict_list,k) for k in key_names}
                
                print(f'Model name: {model_name} file_name:{file_name} test rogue {test_rouge_dict}')
                
                for metric_name, metric_value in test_rouge_dict.items():
                    run.log({"test_"+model_name + '_'+metric_name: metric_value})
                
                val_results[model_name][step_size] = val_rouge_dict
                test_results[model_name][step_size] = test_rouge_dict

                model_val_results[model_name][step_size] = val_rouge_dict
                model_test_results[model_name][step_size] = test_rouge_dict

            # Save results
            with open("Results/"+model_name+"_evaluation_results.json", "w") as f:
                json.dump(model_val_results, f)
            
            # Save results
            with open("Results/"+model_name+"_test_results.json", "w") as f:
                json.dump(model_test_results, f)

            # Save results
            # with open("evaluation_results.json", "w") as f:
            #     json.dump(val_results, f)
            
            # # Save results
            # with open("test_results.json", "w") as f:
            #     json.dump(test_results, f)
    
    return

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception("Usage: model.py model_name")
        
    (_, model_name) = sys.argv
    print(model_name)
    wandb.login(key=chk_config.WANDB_API_KEY)
    run = wandb.init(project=chk_config.WANDB_PROJECT,config={"model":"Val and Test Results"},entity=chk_config .WANDB_ENTITY,group="Val and Test Results")
    checkpoint_results(run,model_name)