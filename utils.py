import torch
from copy import deepcopy as cc

'''
###################################################################################
Save checkpoint at a path
###################################################################################
'''

def save_checkpoint(model, optimizer, scheduler, steps, loss_profile, path):
    checkpoint = {'model' : model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                  'optimizer' : optimizer.state_dict(),
                  'scheduler' : scheduler.state_dict(),
                  'loss_profile' : loss_profile,  
                  'steps' : steps}
    torch.save(checkpoint, path)    
    return

'''
###################################################################################
Load checkpoint from a path
###################################################################################
'''
def load_checkpoint(model, optimizer, scheduler, steps, logging_step_list, tr_running_loss,  
                    val_running_loss, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    steps = scheduler.last_epoch
    logging_step_list = checkpoint['loss_profile']['logging_step_list']
    tr_running_loss = checkpoint['loss_profile']['training_running_loss']
    val_running_loss = checkpoint['loss_profile']['validation_running_loss']
    return steps, logging_step_list, tr_running_loss, val_running_loss


'''
###################################################################################
Truncate the sequence length based on the # of events and 
###################################################################################
'''
def dynamic_batch(batch, device):
    text_tok_src = batch['text_tok_src'].long().to(device)
    text_mask_src = batch['text_mask_src'].long().to(device)
    event_loc_src = batch['event_loc_src'].long().to(device)
    event_mask_src = batch['event_mask_src'].long().to(device)

    text_tok_tar = batch['text_tok_tar'].long().to(device)
    text_mask_tar = batch['text_mask_tar'].long().to(device)


    #==================================================================================#
    #                            DYNAMIC BATCHING - ENCODER                            #
    #==================================================================================#

    max_events_in_batch = event_mask_src.sum(1).max().item() + 3
    event_loc_src =event_loc_src[:, :max_events_in_batch]
    event_mask_src = event_mask_src[:, :max_events_in_batch]

    max_tok_len_in_batch = event_loc_src[:, -1].max().item() + 1

    text_tok_src = text_tok_src[:, :max_tok_len_in_batch]
    text_mask_src = text_mask_src[:, :max_tok_len_in_batch]
    #==================================================================================#


    #==================================================================================#
    #                            DYNAMIC BATCHING - DECODER                            #
    #==================================================================================#
    max_tok_len_in_batch = text_mask_tar.sum(1).max().item()
    text_mask_tar = text_mask_tar[:, :max_tok_len_in_batch]
    text_tok_tar = text_tok_tar[:, :max_tok_len_in_batch]
    #==================================================================================#

    return text_tok_src, text_mask_src, event_loc_src, event_mask_src, text_tok_tar, text_mask_tar

'''
###################################################################################
Config file having the hyper-paramater info for the treebert2bert model
###################################################################################
'''
class Config():
    "Configuration for BERT model"
    vocab_size: int = 30522 # Size of Vocabulary
    dim: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 12 # Numher of Hidden Layers
    n_heads: int = 12 # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    p_drop_hidden: float = 0.1 # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0.1 # Probability of Dropout of Attention Layers
    max_len: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments
    layer_norm_eps: int = 1e-12 # eps value for the LayerNorms
    output_attentions : bool = False # Weather to output the attention scores