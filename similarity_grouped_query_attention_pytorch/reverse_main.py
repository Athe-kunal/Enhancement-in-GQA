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

wandb.login(key=config.WANDB_API_KEY)
run = wandb.init(project=config.WANDB_PROJECT,config={"model":config.MODEL_NAME,"gqa_list":config.GQA_LIST},entity=config.WANDB_ENTITY)

def get_tf_attention_dict(module,kv_heads:int=4):
    transfer_to_gqa: List[str] = ["encoder","decoder","EncDecAttention"]
    tf_attention_dict = defaultdict(list)
    def convert_t5_to_gqa(module, kv_heads: int,similarity_flag:bool=False,inplace: bool = False,curr_name:str=''):
        """Get the list of attention modules based on the flag about encoder, decoder or cross-attention

        Args:
            module: Transformer module/unit
            kv_heads (int): Number of key-value heads
            similarity_flag (bool, optional): Similarity GQA flag. Defaults to False.
            inplace (bool, optional): inplace replace the model with GQA. Defaults to False.

        Returns:
            _type_: _description_
        """
        if isinstance(module, T5Attention) and similarity_flag:
            tf_attention_dict[curr_name].append(module)

        out = module if inplace else deepcopy(module)
        for name, child in out.named_children():
            if name in transfer_to_gqa:
                curr_name = name
                similarity_flag = True
            out._modules[name] = convert_t5_to_gqa(child, kv_heads=kv_heads,similarity_flag=similarity_flag, inplace=True,curr_name=curr_name)
        return out

    convert_t5_to_gqa(module,kv_heads=kv_heads)
    return tf_attention_dict

def get_sim_score(query_heads_attn):
    query_heads_attn_transposed = query_heads_attn.T
    query_heads = torch.tensor_split(query_heads_attn_transposed,8,dim=1)
    sim_keys = ['kv_heads_'+str(i) for i in range(0,len(query_heads)//2)]
    sim_keys_dict = {key: 0.0 for key in sim_keys}
    for i in range(0,len(query_heads)-1,2):
        vec1 = query_heads[i]
        vec2 = query_heads[i+1]
        vec1_flat = vec1.reshape(-1).unsqueeze(0)
        vec2_flat = vec2.reshape(-1).unsqueeze(0)

        # Calculate cosine similarity using PyTorch's F.cosine_similarity
        cosine_sim = F.cosine_similarity(vec1_flat, vec2_flat).item()
        if i==0: dict_name='0'
        elif i==2: dict_name='1'
        elif i==4:dict_name='2'
        elif i==6:dict_name='3'

        sim_keys_dict[f'kv_heads_{dict_name}']=[cosine_sim]
    return sim_keys_dict  

def compute_metrics(eval_batch,tokenizer,metric):
    predictions, labels = eval_batch['input_ids'],eval_batch['labels']
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

def reverse_train(model_name:str=config.MODEL_NAME):
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        model_name
    )

    tokenizer =  AutoTokenizer.from_pretrained(model_name)
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
    tf_attention_dict = get_tf_attention_dict(t5)
    all_similarities_dict = {k:[] for k in config.GQA_LIST}
    val_rouge_dict = {'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [], 'gen_len': []}
    for attn_name in config.GQA_LIST:
        #attn_name is encoder, decoder or cross-attention
        for attn_layer in tf_attention_dict[attn_name]:
            all_similarities_dict[attn_name].append(get_sim_score(attn_layer.q.weight.data))

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
        tf_attention_dict = get_tf_attention_dict(t5)
        curr_similarities_dict = defaultdict(list)

        for attn_name in config.GQA_LIST:
            #attn_name is encoder, decoder or cross-attention
            for attn_layer in tf_attention_dict[attn_name]:
                curr_similarities_dict[attn_name].append(get_sim_score(attn_layer.q.weight.data))
        
        for attn_name,curr_attn_dict in all_similarities_dict.items():

            for idx,attn_layer in enumerate(curr_attn_dict):
                for attn_heads,sim_scores in attn_layer.items():
                    curr_val = curr_similarities_dict[attn_name][idx][attn_heads]
                    all_similarities_dict[attn_name][idx][attn_heads].extend(curr_val)
        
        eval_dict_list = []
        for eval_batch in eval_dataloader:
            eval_dict_list.append(compute_metrics(eval_batch,tokenizer,metric))
        
        key_names = eval_dict_list[0].keys()
        average_dict = {k:get_avg(eval_dict_list,k) for k in key_names}
        for k in average_dict.keys():
            val_rouge_dict[k].append(average_dict[k])
    wandb.log({"val_rouge":val_rouge_dict})

    #Plotting
    plt.style.use('ggplot')

    fig, ax = plt.subplots(3, 6, sharex=True,figsize=(30, 20))

    fax = ax.ravel()
    for i in range(6):
        for j in range(4):
            fax[i].plot(all_similarities_dict['encoder'][i][f'kv_heads_{j}'],label=f'kv_heads_{j}')
        fax[i].set_xlabel("Epochs")
        fax[i].set_title(f'Layer {i}')
    fax[0].set_ylabel("ENCODER Similarity Score")
    fax[0].legend()
        
    for i in range(6,12):
        for j in range(4):
            fax[i].plot(all_similarities_dict['decoder'][i-6][f'kv_heads_{j}'],label=f'kv_heads_{j}')
        fax[i].set_xlabel("Epochs")
        fax[i].set_title(f'Layer {i-6}')
    fax[6].set_ylabel("DECODER Similarity Score")
    fax[6].legend()

    for i in range(12,18):
        for j in range(4):
            fax[i].plot(all_similarities_dict['EncDecAttention'][i-12][f'kv_heads_{j}'],label=f'kv_heads_{j}')
        fax[i].set_xlabel("Epochs")
        fax[i].set_title(f'Layer {i-12}')
    fax[12].set_ylabel("CROSS-ATTENTION Similarity Score")
    fax[12].legend()
    t5.eval()

    wandb.log({f"{config.MODEL_NAME}_SIM_DICT":all_similarities_dict})
    wandb.log({f"{config.MODEL_NAME}_PLOT":fig})
    test_dict_list = []
    for test_batch in test_dataloader:
        test_dict_list.append(compute_metrics(test_batch,tokenizer,metric))
    
    key_names = test_dict_list[0].keys()
    test_rouge_dict = {k:get_avg(test_dict_list,k) for k in key_names}
    wandb.log({"test_rouge":test_rouge_dict})

    return all_similarities_dict,val_rouge_dict,test_rouge_dict

    

if __name__ == '__main__':
    reverse_train()








