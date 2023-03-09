import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy as cc
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.nn import GELU as GELU
from dataset import *
from TreeBERT_attention import *
import pdb
INFINITY = 1e15
EPSILON = 1e-15

'''===============================================================================
                            EMBEDDING LAYER
==============================================================================='''
class BertEmbeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        self.word_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim, padding_idx=0) # token embedding
        self.position_embeddings = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        self.token_type_embeddings = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.LayerNorm = nn.LayerNorm(cfg.dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.word_embeddings(x) + self.position_embeddings(pos) + self.token_type_embeddings(seg)
        e = self.LayerNorm(e)
        e = self.dropout(e)
        return e


'''===============================================================================
                            FORWARD EXPANSION LAYER
==============================================================================='''

# f(X) -> Linear(D => 4xD) -> GELU() -> X2
class BertIntermediate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.dim, cfg.dim_ff)
        self.GELU = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.GELU(hidden_states)
        return hidden_states

# f(X) -> Linear(4xD => D) -> Drouput -> X1
#  X, f(X) -> LayerNorm( X + X1 )
class BertOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.dim_ff, cfg.dim)
        self.LayerNorm = nn.LayerNorm(cfg.dim, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


'''===============================================================================
                            BERT ENCODER LAYER
==============================================================================='''
class BertLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.group_attention = GroupAttention2(cfg)
        self.attention = BertAttention(cfg)
        self.intermediate = BertIntermediate(cfg)
        self.output = BertOutput(cfg)

    def forward(self, hidden_states, ip_event_loc, ip_event_mask, attention_mask=None, output_attentions = False, A_prior = None):
        C_prior = None

        if A_prior != None:
            # C_prior, A_prior = self.group_attention(hidden_states, attention_mask, A_prior)
            C_prior, A_prior = self.group_attention(hidden_states = hidden_states, ip_event_loc = ip_event_loc, 
                                                    attention_mask = attention_mask, ip_event_mask = ip_event_mask, prior_A = A_prior)

        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions, C_prior)
        intermediate_output = self.intermediate(self_attention_outputs['hidden_states'])
        layer_output = self.output(intermediate_output, self_attention_outputs['hidden_states'])

        outputs = {'hidden_states':layer_output, 'attn_scores': self_attention_outputs['attn_scores'], 
                  'A_prior': A_prior, 'C_prior': C_prior}
        return outputs


'''===============================================================================
                                    Tree-BERT
==============================================================================='''

class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(cfg) for _ in range(cfg.n_layers)])

    def forward(self, hidden_states, ip_event_loc, ip_event_mask, attention_mask=None, output_attentions = False, 
                output_hidden_states = False, A_prior = None, output_A_prior = False, output_C_prior = False):   

        all_hidden_states = [] if output_hidden_states else None
        all_self_attentions = [] if output_attentions else None
        all_A_priors = [] if output_A_prior else None
        all_C_priors = [] if output_C_prior else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            layer_outputs = layer_module(hidden_states, ip_event_loc, ip_event_mask, attention_mask, output_attentions, A_prior)
            hidden_states, attn_scores = layer_outputs['hidden_states'], layer_outputs['attn_scores'], 
            C_prior, A_prior = layer_outputs['C_prior'], layer_outputs['A_prior']

            if output_attentions:
                all_self_attentions.append(attn_scores)
            if output_A_prior:
                all_A_priors.append(A_prior)
            if output_C_prior:
                all_C_priors.append(C_prior)

        outputs = {'hidden_states': hidden_states, 'attn_scores': all_self_attentions, 
                  'all_hidden_states': all_hidden_states, 'all_A_priors': all_A_priors, 'all_C_priors': all_C_priors}
        return outputs 


'''===============================================================================
                                Get the initial value of A_matrix 
==============================================================================='''
class Tree_BERT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
    
    def forward(self, x, seg, ip_event_loc, ip_event_mask, attention_mask=None, output_attentions = False, 
                output_hidden_states = False, A_prior = None, output_A_prior = False, output_C_prior = False):

        embedding_output = self.embeddings(x, seg)
        encoded_output = self.encoder(hidden_states = embedding_output, 
                                      attention_mask = attention_mask, 
                                      ip_event_loc = ip_event_loc, 
                                      ip_event_mask = ip_event_mask,
                                      output_attentions = output_attentions, 
                                      output_hidden_states = output_hidden_states,
                                      A_prior = A_prior,
                                      output_A_prior = output_A_prior,
                                      output_C_prior = output_C_prior)

        return encoded_output