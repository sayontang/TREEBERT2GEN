import torch
from torch import nn
import numpy as np
from copy import deepcopy as cc
import torch.nn.functional as F
INFINITY = 1e15
EPSILON = 1e-15

class GroupAttention2(nn.Module):
    def __init__(self, cfg):
        super(GroupAttention2, self).__init__()
        self.d_model = cfg.dim
        self.linear_key = nn.Linear(cfg.dim, cfg.dim)
        self.linear_query = nn.Linear(cfg.dim, cfg.dim)
        self.LayerNorm = nn.LayerNorm(cfg.dim, eps= cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.p_drop_attn)

        torch.nn.init.xavier_normal_(self.linear_key.weight)
        torch.nn.init.zeros_(self.linear_key.bias)

        torch.nn.init.xavier_normal_(self.linear_query.weight)
        torch.nn.init.zeros_(self.linear_query.bias)

        torch.nn.init.uniform_(self.LayerNorm.weight)
        torch.nn.init.zeros_(self.LayerNorm.bias)

    def forward(self, hidden_states, attention_mask, ip_event_loc, ip_event_mask, prior_A):
        ''' 
        hidden_states = B x S x D
        attention_mask = B x S(V/K)
        A_prior = B x S(Q) x S(V/K)
        '''

        B, S_full = hidden_states.size()[:2]
        hidden_event_subset = hidden_states[np.arange(B)[:, None], ip_event_loc]
        context = self.LayerNorm(hidden_event_subset)
        B, S = hidden_event_subset.size()[:2]

        a = torch.from_numpy(np.diag(np.ones(S - 1, dtype=np.int32),1)).to(hidden_states.device)
        b = torch.from_numpy(np.diag(np.ones(S, dtype=np.int32),0)).to(hidden_states.device)
        c = torch.from_numpy(np.diag(np.ones(S - 1, dtype=np.int32),-1)).to(hidden_states.device)
        tri_matrix = torch.from_numpy(np.triu(np.ones([S, S], dtype=np.float32),0)).to(hidden_states.device)

        a_full = torch.from_numpy(np.diag(np.ones(S_full - 1, dtype=np.int32),1)).to(hidden_states.device)
        b_full = torch.from_numpy(np.diag(np.ones(S_full, dtype=np.int32),0)).to(hidden_states.device)
        c_full = torch.from_numpy(np.diag(np.ones(S_full - 1, dtype=np.int32),-1)).to(hidden_states.device)
        tri_matrix_full = torch.from_numpy(np.triu(np.ones([S_full, S_full], dtype=np.float32),0)).to(hidden_states.device)

        # mask -> BxSxS
        mask = ip_event_mask[:, None, :] & (a+c)[None, :, :]
        
        key = self.linear_key(context)
        query = self.linear_query(context)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model / 2)
        scores = scores.masked_fill(mask == 0, (-1 * INFINITY))

        # Compute the A for the [et_sep] locations 
        # A -> B x S x S
        A = F.softmax(scores, dim=-1)
        A = torch.sqrt(A * A.transpose(-2,-1) + EPSILON)

        # A -> B x (S-1) ; Extract the diagonal(offset 1) elements from the A
        A = torch.cat([torch.diagonal(A[batch], 1)[None, :] for batch in range(B)])

        # scatter the above A matrix in the A_full (B x S_full) matrix
        A_new = cc(attention_mask).float()
        A_new = A_new.scatter(1, ip_event_loc[:, :-1], A.float())

        A_new = A_new[:, :-1]
        A_new= [torch.diag(A_new[batch, :], 1)[None, :, :] for batch in range(B)]
        A_new = torch.cat(A_new)
        A_new += A_new.transpose(-1, -2).contiguous()
        A_new = prior_A + (1. - prior_A)*A_new

        t = torch.log(A_new + EPSILON).masked_fill(a_full==0, 0).matmul(tri_matrix_full)
        C_prior = tri_matrix_full.matmul(t).exp().masked_fill((tri_matrix_full.int()-b_full)==0, 0)
        C_prior = C_prior + C_prior.transpose(-2, -1).contiguous() + torch.from_numpy(np.diag(np.ones(S_full))).to(hidden_states.device)
        
        return C_prior.float(), A_new.float()


# X -> self_attn -> X [calculated by self attention]
class BertSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Linear(cfg.dim, cfg.dim)
        self.key = nn.Linear(cfg.dim, cfg.dim)
        self.value = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = nn.Dropout(cfg.p_drop_attn)
        self.n_heads = cfg.n_heads

    def forward(self, x, attention_mask = None, output_attentions = False, C_prior = None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        attention_mask_dim = B x S (dim S is for every value in the "x"), so need to repeat it for every query
        """
        B, S, D = x.shape
        H = self.n_heads
        W = int( D/H )
        assert W * H == D

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.query(x), self.key(x), self.value(x)
        q, k, v = q.reshape((B, S, H, W)), k.reshape((B, S, H, W)), v.reshape((B, S, H, W))
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # (B, H, S(Q), W) @ (B, H, W, S(K/V)) -> (B, H, S(Q), S(K/V)) -Masking -> softmax-> (B, H, S(Q), S(K/V))
        attn_scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        
        # Set the attention score at places where MASK = 0 to very low value (-1e9)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask[:, None, None, :] == 0, (-1 * INFINITY))
        attn_scores = self.dropout(F.softmax(attn_scores, dim=-1))
    
        # C_prior -> B x S x S
        if C_prior != None:
            attn_scores = attn_scores * C_prior[:, None, :, :]

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
        hidden_states = (attn_scores @ v).transpose(1, 2).contiguous()
        hidden_states = hidden_states.reshape(B, S, D)

        result = {}
        result['hidden_states'] = hidden_states 
        result['attn_scores'] =  None
        if output_attentions :
            result['attn_scores'] =  attn_scores

        return result


#  f(X) -> Linear (D => D) -> Dropout -> X1
#  X, f(X) -> LayerNorm( X + X1 )
class BertSelfOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dense = nn.Linear(cfg.dim, cfg.dim)
        self.LayerNorm = nn.LayerNorm(cfg.dim, eps= cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# X -> self_attn -> f(X) -> Linear (D => D) -> Dropout -> X1
# Y = LayerNorm( X + X1 )
class BertAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self = BertSelfAttention(cfg)
        self.output = BertSelfOutput(cfg)

    def forward(self, hidden_states, attention_mask=None, output_attentions = False, C_prior = None):
        self_outputs = self.self(hidden_states,  attention_mask, output_attentions, C_prior)
        attention_output = self.output(self_outputs['hidden_states'], hidden_states)

        outputs = {'hidden_states': attention_output, 'attn_scores': self_outputs['attn_scores']}
        return outputs     