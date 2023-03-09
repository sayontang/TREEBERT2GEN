import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as cc

class Config():
    "Configuration for BERT model"
    vocab_size: int = 30522 # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    layer_norm_eps: int = 1e-12 # eps value for the LayerNorms
    output_attentions : bool = False # Weather to output the attention scores


'''=========================================================
                    BERT emebdding layer
========================================================='''

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
        return 
        

'''=========================================================
                 BERT self-attn layer
========================================================='''
        
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

    def forward(self, x, mask = None, output_attentions = False):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        B, S, D = x.shape
        H = self.n_heads
        W = int( D/H )
        assert W * H == D

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.query(x), self.key(x), self.value(x)
        q, k, v = q.reshape((B, S, H, W)), k.reshape((B, S, H, W)), v.reshape((B, S, H, W))
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -Masking -> softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        
        # Set the attention score at places where MASK = 0 to very low value (-1e9)
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores = scores.masked_fill(mask == 0, -1e20)
        scores = self.dropout(F.softmax(scores, dim=-1))
        
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W) -merge-> (B, S, D)
        hidden_states = (scores @ v).transpose(1, 2).contiguous()
        hidden_states = hidden_states.reshape(B, S, D)
        return (hidden_states, scores) if output_attentions else (hidden_states,)


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

    def forward(self, hidden_states, attention_mask=None, output_attentions = False):
        self_outputs = self.self(hidden_states,  attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)

        # add attentions if we output them; self_outputs[1:] is the attn score, attention_output is the hidden representation
        outputs = (attention_output,) + self_outputs[1:] if output_attentions else (attention_output,)


'''=========================================================
                BERT fwd-expansion layer
========================================================='''

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


'''=========================================================
                BERT single encoder layer
========================================================='''

class BertLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = BertAttention(cfg)
        self.intermediate = BertIntermediate(cfg)
        self.output = BertOutput(cfg)

    def forward(self, hidden_states, attention_mask=None, output_attentions = False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs if output_attentions else (layer_output,)
        return outputs

'''=========================================================
                        BERT encoder
========================================================='''

class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(cfg) for _ in range(cfg.n_layers)])

    def forward(self, hidden_states, attention_mask=None, output_attentions = False, output_hidden_states = False):        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        outputs = (hidden_states,) 
        outputs = outputs + (all_hidden_states,) + (all_self_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


'''=========================================================
                        BERT model
========================================================='''

class my_BERT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.encoder = BertEncoder(cfg)
    
    def forward(self, x, seg, attention_mask=None, output_attentions = False, output_hidden_states = False):

        embedding_output = self.embeddings(x, seg)
        encoded_output = self.encoder(hidden_states = embedding_output, 
                                      attention_mask = attention_mask, 
                                      output_attentions = output_attentions, 
                                      output_hidden_states = output_hidden_states)