from transformers import T5ForConditionalGeneration, T5Tokenizer,T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5Config, T5Block
from copy import deepcopy
from typing import List, Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5LayerCrossAttention
from collections import defaultdict
from config import *
from t5_SGQA import CustomT5SelfAttention



def create_t5_config_from_block(block):
    # Create a T5Config object with necessary parameters from the block
    config = T5Config(
        is_decoder=block.is_decoder,
        d_model=block.d_model,
        relative_attention_num_buckets = block.relative_attention_num_buckets,
        relative_attention_max_distance = block.relative_attention_max_distance,
        d_kv= block.key_value_proj_dim,
        num_heads=block.n_heads,
        dropout_rate=block.dropout,
    )
    return config

def mean_pool(key_or_value_heads,d_model,n_heads,kv_heads,h_dim):
    '''
    Please note that weight data is transpose of defined layer
    For example, k = nn.Linear(in_features=512,out_features=256) will give shape of (256,512)
    '''
    #(inner_dim, d_model) -> (n_heads(8), h_dim(64),d_model)
    key_or_value_heads = key_or_value_heads.view(n_heads,h_dim,d_model)
    
    #(n_heads(8), h_dim(64),d_model) -> (#kv_heads, #headspergroup, h_dim(64),d_model) and mean along axis 1
    key_or_value_heads = key_or_value_heads.view(kv_heads,n_heads//kv_heads,h_dim,d_model).mean(axis=1)
    
    #reshape to (h_dim*kv_heads,d_model)
    key_or_value_heads = key_or_value_heads.view(kv_heads*h_dim,d_model)
    return key_or_value_heads

def add_pool(key_or_value_heads,d_model,n_heads,kv_heads,h_dim):
    '''
    Please note that weight data is transpose of defined layer
    For example, k = nn.Linear(in_features=512,out_features=64) will give shape of (64,512)
    '''
    #(inner_dim (512), d_model) -> (n_heads(8), h_dim(64),d_model)
    # print(key_or_value_heads.shape)
    key_or_value_heads = key_or_value_heads.view(n_heads,h_dim,d_model)
    
    #(n_heads(8), h_dim(64),d_model) -> (#kv_heads, #headspergroup, h_dim(64),d_model) and add along axis 1
    key_or_value_heads = key_or_value_heads.view(kv_heads,n_heads//kv_heads,h_dim,d_model).sum(axis=1)
    
    #reshape to (h_dim*kv_heads,d_model)
    key_or_value_heads = key_or_value_heads.view(kv_heads*h_dim,d_model)
    return key_or_value_heads

class WT5GQA(nn.Module):
    def __init__(self,
        is_decoder: bool,
        d_model: int,
        key_value_proj_dim: int,
        n_heads: int,
        kv_heads: int,
        dropout: float,
        has_relative_attention_bias: bool,
        relative_attention_num_buckets: int,
        relative_attention_max_distance: int):

        super(WT5GQA,self).__init__()
        self.is_decoder = is_decoder
        self.d_model = d_model
        self.key_value_proj_dim = key_value_proj_dim
        self.n_heads = n_heads
        self.kv_heads = kv_heads
        self.dropout = dropout

        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.kv_dim = self.kv_heads * self.key_value_proj_dim
        self.kv_heads = kv_heads
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        if IF_RANDOM:
            self.params = nn.ParameterDict({
                f"key": nn.Parameter(torch.randn((self.n_heads,1))),
                f"value": nn.Parameter(torch.randn((self.n_heads,1))),
                })
        else:
            self.params = nn.ParameterDict({
                f"key": nn.Parameter(torch.full((self.n_heads,1),0.5)),
                f"value": nn.Parameter(torch.full((self.n_heads,1),0.5)),
                })
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()  # type: ignore
        self.gradient_checkpointing = False

        self._relative_position_bucket = T5Attention._relative_position_bucket
    
    @classmethod
    def from_t5_attention(cls, t5: T5Attention, kv_heads: int):
        t5_wgqa = WT5GQA(
            is_decoder=t5.is_decoder,
            d_model=t5.d_model,
            key_value_proj_dim=t5.key_value_proj_dim,
            n_heads=t5.n_heads,
            kv_heads=kv_heads,
            dropout=t5.dropout,
            has_relative_attention_bias=t5.has_relative_attention_bias,
            relative_attention_num_buckets=t5.relative_attention_num_buckets,
            relative_attention_max_distance=t5.relative_attention_max_distance
        )

        # Copy all of the weights verbatim from the original T5Attention module.
        # NOTE: In the T5 GQA implementation, all of the attention head aggregations
        # happen in the 'forward' method.  The weights themselves are not modified.
        t5_wgqa.q.weight.data = t5.q.weight.data
        t5_wgqa.k.weight.data = t5.k.weight.data
        t5_wgqa.v.weight.data = t5.v.weight.data
        t5_wgqa.o.weight.data = t5.o.weight.data

        # if IF_RANDOM:
        #     params = nn.ParameterDict({
        #         f"key": nn.Parameter(torch.randn((t5.n_heads,1))),
        #         f"value": nn.Parameter(torch.randn((t5.n_heads,1))),
        #         })
        # else:
        #     params = nn.ParameterDict({
        #         f"key": nn.Parameter(torch.full((t5.n_heads,1),0.5)),
        #         f"value": nn.Parameter(torch.full((t5.n_heads,1),0.5)),
        #         })


        # t5_wgqa_k_mod = t5_wgqa_k_mod.view(t5.d_model,t5.n_heads,t5.d_model//t5.n_heads)
        # t5_wgqa_k_mod = torch.multiply(t5_wgqa_k_mod,params[f"key"])
        # t5_wgqa_v_mod = t5_wgqa_v_mod.view(t5.d_model,t5.n_heads,t5.d_model//t5.n_heads)
        # t5_wgqa_v_mod = torch.multiply(t5_wgqa_v_mod,params[f"value"])
        
        # '''added part'''
        # # x.permute(*torch.arange(x.ndim - 1, -1, -1))
        # t5_wgqa_k_mod = torch.reshape(t5_wgqa_k_mod,(t5.d_model,-1))
        # t5_wgqa_v_mod = torch.reshape(t5_wgqa_v_mod,(t5.d_model,-1))
        # modified_k_weights =  torch.nn.Parameter(t5_wgqa_k_mod.T)
        # t5_wgqa.k = nn.Linear(in_features=t5.d_model, out_features=t5.d_model,bias=False)
        # t5_wgqa.k.weight.data = modified_k_weights

        # # x.permute(*torch.arange(x.ndim - 1, -1, -1))
        # modified_v_weights = torch.nn.Parameter(t5_wgqa_v_mod.T)
        # t5_wgqa.v = nn.Linear(in_features = t5.d_model, out_features = t5.d_model,bias=False)
        # t5_wgqa.v.weight.data = modified_v_weights
        '''added part ended'''

        if t5.has_relative_attention_bias:
            t5_wgqa.relative_attention_bias.weight.data = (
                t5.relative_attention_bias.weight.data
            )

        return t5_wgqa

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            # NOTE: Changed from the original definition in T5Attention.
            sequence_length = states.shape[1]
            return states.view(batch_size, sequence_length, -1, self.key_value_proj_dim).transpose(1, 2)
            # return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
             # NOTE: Changed from the original definition in T5Attention.
            sequence_length = states.shape[2]
            return (states.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1))
            # return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        # print(self.k) #Linear layer with (512,512)
        # print(query_states.shape)
        #(512,512) key matrix --> 8 x (512,64) (after taking transpose) (512,8,64)--> (512,4,2,64) --> add pool 3rd dimension --> (512,4,1,64) 
        # (512,4,2,64) repeat interleave
        #After taking projection
        # (batch_size, seq_lenth,512) @ (512,256) --> (batch_size,seq_length,256) --> (batch_size,seq_length,4,64) --> (batch_size,seq_length,8,64)

        #k- (512,256)  --> (512,512)
        # self.k_agg = add_pool(key_states,self.d_model,self.n_heads,self.kv_heads,self.d_model//self.n_heads)
        # self.v_agg = add_pool(value_states,self.d_model,self.n_heads,self.kv_heads,self.d_model//self.n_heads)
        k_weight_data = self.k.weight.data.T
        k_weight_data = k_weight_data.view(self.d_model,self.n_heads,self.d_model//self.n_heads)
        k_weight_data = torch.multiply(k_weight_data,self.params[f"key"])
        k_weight_data = torch.reshape(k_weight_data,(self.d_model,-1)).T

        k_weight_data = add_pool(k_weight_data,d_model=self.d_model,n_heads=self.n_heads,kv_heads=self.kv_heads,h_dim=self.d_model//self.n_heads) #(256,512)
        k_weight_data = k_weight_data.view(self.kv_heads,self.d_model//self.n_heads,self.d_model).repeat_interleave(self.n_heads//self.kv_heads,dim=0)
        k_weight_data = torch.reshape(k_weight_data,(-1,self.d_model))
        self.k.weight.data = k_weight_data
        # print(k_weight_data.shape)
        v_weight_data = self.v.weight.data.T
        v_weight_data = v_weight_data.view(self.d_model,self.n_heads,self.d_model//self.n_heads)
        v_weight_data = torch.multiply(v_weight_data,self.params[f"value"])
        v_weight_data = torch.reshape(v_weight_data,(self.d_model,-1)).T
        v_weight_data = add_pool(self.v.weight.data,d_model=self.d_model,n_heads=self.n_heads,kv_heads=self.kv_heads,h_dim=self.d_model//self.n_heads) #(256,512)
        v_weight_data = v_weight_data.view(self.kv_heads,self.d_model//self.n_heads,self.d_model).repeat_interleave(self.n_heads//self.kv_heads,dim=0)
        v_weight_data = torch.reshape(v_weight_data,(-1,self.d_model))
        self.v.weight.data = v_weight_data

        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        '''Note added this part'''
        #repeat key value states
        # key_states = key_states.repeat_interleave(self.n_heads//self.kv_heads,dim=1)

        # value_states = value_states.repeat_interleave(self.n_heads//self.kv_heads,dim=1)

        # compute scores
        # print(key_states.shape)
        
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2) #changed here
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = T5Attention.compute_bias(self,real_seq_length, key_length, device=scores.device)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask
        # value_states = add_pool(value_states,self.d_model,self.n_heads,self.kv_heads,self.d_model//self.n_heads)
        attn_output = unshape(torch.matmul(attn_weights, value_states)) # (batch_size, seq_length, dim) #changed here
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs



def convert_t5_to_wgqa(module, kv_heads: int,weight_flag:bool=False,inplace: bool = False,curr_name:str=''):
    """Get the list of attention modules based on the flag about encoder, decoder or cross-attention

    Args:
        module: Transformer module/unit
        kv_heads (int): Number of key-value heads
        similarity_flag (bool, optional): Similarity GQA flag. Defaults to False.
        inplace (bool, optional): inplace replace the model with GQA. Defaults to False.

    Returns:
        _type_: _description_
    """
    if isinstance(module, T5Attention) and weight_flag:
        # print(curr_name)
        return WT5GQA.from_t5_attention(module, kv_heads=kv_heads)

    out = module if inplace else deepcopy(module)
    for name, child in out.named_children():
        if name in GQA_LIST:
            curr_name = name
            weight_flag = True
        out._modules[name] = convert_t5_to_wgqa(child, kv_heads=kv_heads,weight_flag=weight_flag, inplace=True,curr_name=curr_name)
    return out
    


if __name__=='__main__':
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
    summ_text = '''After the sound and the fury, weeks of demonstrations and anguished calls for racial justice, t
                he man whose death gave rise to an international movement, and whose last words — “I can’t breathe” — have 
                been a rallying cry, will be laid to rest on Tuesday at a private funeral in Houston.George Floyd, who was 46, 
                will then be buried in a grave next to his mother’s.The service, scheduled to begin at 11 a.m. at the Fountain of 
                Praise church, comes after five days of public memorials in Minneapolis, North Carolina and Houston and two weeks 
                after a Minneapolis police officer was caught on video pressing his knee into Mr. Floyd’s neck for nearly nine minutes 
                before Mr. Floyd died. That officer, Derek Chauvin, has been charged with second-degree murder and second-degree 
                manslaughter. His bail was set at $1.25 million in a court appearance on Monday. The outpouring of anger and outrage 
                after Mr. Floyd’s death — and the speed at which protests spread from tense, chaotic demonstrations in the city where 
                he died to an international movement from Rome to Rio de Janeiro — has reflected the depth of frustration borne of 
                years of watching black people die at the hands of the police or vigilantes while calls for change went unmet.', 80'''

    input_ids = tokenizer("summarize: "+summ_text, return_tensors="pt").input_ids
    outputs = t5.generate(input_ids, max_new_tokens=128)
    text = tokenizer.batch_decode(outputs[0], skip_special_tokens=False)
    print(f'Generated text with pretrained model: {text}')
    #convert t5 to gqa
    for kv_heads in [8]:
        t5_wgqa = convert_t5_to_wgqa(t5,kv_heads=kv_heads,inplace=False)
        # print(t5_wgqa)
        t5_wgqa.eval()

        outputs = t5_wgqa.generate(input_ids, max_new_tokens=128)
        text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
        print(f'Generated text with kv_heads:{kv_heads}: {text}')