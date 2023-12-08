from transformers import T5ForConditionalGeneration, T5Tokenizer,T5Config
from transformers.models.t5.modeling_t5 import T5Attention, T5Config, T5Block
from copy import deepcopy
from typing import List, Optional
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5Attention, T5LayerSelfAttention, T5LayerCrossAttention

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
    For example, k = nn.Linear(in_features=512,out_features=64) will give shape of (64,512)
    '''
    #(inner_dim (512), d_model) -> (n_heads(8), h_dim(64),d_model)
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

class WeightT5SelfAttention(T5Attention):
    def __init__(self, t5_decoder_attention_block,kv_heads:int,weight_flag:bool=False,if_random:bool=False):
        config = create_t5_config_from_block(t5_decoder_attention_block)
        super(WeightT5SelfAttention,self).__init__(config)

        # Transfer complete layers from the provided T5 attention block
        self.q = t5_decoder_attention_block.q
        self.k = t5_decoder_attention_block.k
        self.v = t5_decoder_attention_block.v
        self.o = t5_decoder_attention_block.o
        self.has_relative_attention_bias = t5_decoder_attention_block.has_relative_attention_bias
        self.kv_heads = kv_heads
        self.weight_flag = weight_flag
        self.grouped_pairs = None
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = t5_decoder_attention_block.relative_attention_bias
        self.pruned_heads = t5_decoder_attention_block.pruned_heads
        self.gradient_checkpointing = t5_decoder_attention_block.gradient_checkpointing
        self.if_random = if_random
        # if if_random:
        # self.params = nn.ParameterDict({
        #     f"key": nn.Parameter(torch.full((self.n_heads,1))),
        #     f"value": nn.Parameter(torch.full((self.n_heads,1))),
        #         })
        # else:
        self.params = nn.ParameterDict({
            "wk": nn.Parameter(torch.full((self.n_heads,1),0.5)),
            "wv": nn.Parameter(torch.full((self.n_heads,1),0.5))
        })

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
        # print(self.params["key"].data.shape)
        k_weight_data = torch.transpose(self.k.weight,0,1)
        #(512,8,64)
        k_weight_data = k_weight_data.view(self.d_model,self.n_heads,self.d_model//self.n_heads)
        # print(k_weight_data)
        k_weight_data = torch.multiply(k_weight_data,self.params["wk"])
        # print(k_weight_data)
        # return 
        k_weight_data = torch.transpose(torch.reshape(k_weight_data,(self.d_model,-1)),0,1)

        k_weight_data = add_pool(k_weight_data,d_model=self.d_model,n_heads=self.n_heads,kv_heads=self.kv_heads,h_dim=self.d_model//self.n_heads) #(256,512)
        k_weight_data = k_weight_data.view(self.kv_heads,self.d_model//self.n_heads,self.d_model).repeat_interleave(self.n_heads//self.kv_heads,dim=0)
        k_weight_data = torch.reshape(k_weight_data,(-1,self.d_model))
        # print(k_weight_data.shape)
        # self.k.weight.data = k_weight_data
        k1 = nn.Linear(in_features=512,out_features=512,bias=False) 
        k1.weight.data = k_weight_data
        # print(k_weight_data.shape)
        v_weight_data = torch.transpose(self.v.weight,0,1)
        v_weight_data = v_weight_data.view(self.d_model,self.n_heads,self.d_model//self.n_heads)
        v_weight_data = torch.multiply(v_weight_data,self.params["wv"])
        v_weight_data = torch.transpose(torch.reshape(v_weight_data,(self.d_model,-1)),0,1)
        v_weight_data = add_pool(self.v.weight.data,d_model=self.d_model,n_heads=self.n_heads,kv_heads=self.kv_heads,h_dim=self.d_model//self.n_heads) #(256,512)
        v_weight_data = v_weight_data.view(self.kv_heads,self.d_model//self.n_heads,self.d_model).repeat_interleave(self.n_heads//self.kv_heads,dim=0)
        v_weight_data = torch.reshape(v_weight_data,(self.d_model,-1))
        v1 = nn.Linear(in_features=512,out_features=512,bias=False)
        v1.weight.data = v_weight_data
        # get key/value states
        key_states = project(
            hidden_states, k1, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, v1, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        '''Note added this part'''
        #repeat key value states
        # key_states = key_states.repeat_interleave(self.n_heads//self.kv_heads,dim=1)

        # value_states = value_states.repeat_interleave(self.n_heads//self.kv_heads,dim=1)

        # compute scores
        # print(key_states.shape)
        # print(value_states.shape)
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
                position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

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

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim) #changed here
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs



def convert_t5_to_wgqa(module, kv_heads: int,weight_flag:bool=False,inplace: bool = False,if_random:bool=False):
    
    out = module if inplace else deepcopy(module)
    
    # use custom attention blocks in decoder
    for layer in out.decoder.block:
        layer.layer[0].SelfAttention = WeightT5SelfAttention(layer.layer[0].SelfAttention,kv_heads,weight_flag,if_random=if_random)
        layer.layer[1].EncDecAttention  = WeightT5SelfAttention(layer.layer[1].EncDecAttention,kv_heads,weight_flag,if_random=if_random)

    return out


if __name__=='__main__':
    t5: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
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

    input_ids = tokenizer("summarize:"+summ_text, return_tensors="pt").input_ids
    outputs = t5.generate(input_ids, max_new_tokens=128)
    text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
    print(f'Generated text with pretrained model: {text}')
    #convert t5 to gqa
    for kv_heads in [4]:
        for weight_flag in [True]:
            t5_gqa = convert_t5_to_wgqa(t5,kv_heads=kv_heads, weight_flag=weight_flag, inplace=False)
            print(t5_gqa)
            t5_gqa.eval()

            outputs = t5_gqa.generate(input_ids, max_new_tokens=128)
            text = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
            print(f'Generated text with kv_heads:{kv_heads} and weight flag:{weight_flag} : {text}')