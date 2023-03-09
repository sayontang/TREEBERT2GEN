import torch
from transformers import  BertModel, EncoderDecoderModel,AdamW, get_linear_schedule_with_warmup
import argparse
from copy import deepcopy as cc
from dataset import *
from tqdm import tqdm
import time
import pdb
import os
import numpy as np
from TreeBERT import *
from torch import nn
import pdb
INFINITY = 1e15
EPSILON = 1e-15
'''
###################################################################################
Defining the Tree2BERT class
###################################################################################
'''
class TreeBERT2BERT(nn.Module):

    def __init__(self, config):
        super(TreeBERT2BERT, self).__init__()
        self.config = config
        self.encoder = Tree_BERT(self.config)

        BERT = BertModel.from_pretrained('bert-base-uncased')
        self.encoder.load_state_dict(BERT.state_dict(), strict = False);

        del BERT
        self.decoder = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased').decoder
    
    def forward(self, src_text_tok, src_text_mask, src_event_loc, src_event_pad_mask, tar_text_tok, tar_pad_mask):
        

        # indices with label -100 will be ignored in the loss computation
        decoder_label = cc(tar_text_tok)
        decoder_label[tar_text_tok == 0] = -100

        # Get the encoded input
        A_initial = self.get_initial_A(ip_text_mask = src_text_mask, ip_event_loc = src_event_loc)
        encoded_src = self.encoder(x = src_text_tok, 
                                   seg = torch.ones_like(src_text_tok), 
                                   ip_event_loc = src_event_loc, 
                                   ip_event_mask = src_event_pad_mask, 
                                   attention_mask = src_text_mask, 
                                   output_attentions = False, 
                                   output_hidden_states = False, 
                                   A_prior = A_initial, 
                                   output_A_prior = False,
                                   output_C_prior = True)
        
        # Pass the encoded input to the decoder
        decoder_out = self.decoder(input_ids = tar_text_tok,
                            attention_mask = tar_pad_mask,
                            encoder_hidden_states = encoded_src['hidden_states'],
                            encoder_attention_mask = src_text_mask,
                            labels = decoder_label)
        return  decoder_out

    def get_initial_A(self, ip_text_mask, ip_event_loc):

        A_initial = cc(ip_text_mask)*1.0

        # Set all the locations with [et_sep] tokens as 0.0
        A_initial = A_initial.scatter(1, ip_event_loc.long(), torch.zeros_like(ip_event_loc).float())

        # exclude the last token as this vector will be a diagonal with offset 1
        A_initial = A_initial[:, :-1]

        # Create a diagonal matrix with this A as the diagonal at offset 1
        A_initial= [torch.diag(A_initial[i, :].float(), 1)[None, :, :] for i in range(A_initial.shape[0])]
        A_initial = torch.cat(A_initial)

        # Add it's transpose, as the A matrix is supposed to be symmetric
        A_initial += A_initial.transpose(-1, -2).contiguous()
        return A_initial

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


'''
###################################################################################
Defining the argument parser
###################################################################################
'''
def parse():
    parser = argparse.ArgumentParser(description = "Conditional event sequence generation with Tree transformer as encoder and BERT as decoder.")
    parser.add_argument('-max_seq_len', type=int, default = 256, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_ev_len', type=int, default = 20, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_steps', type = int, default= 5000, help="Number of epochs")
    parser.add_argument('-batch_size', type = int, default= 2, help="Batch size")
    parser.add_argument('-load_checkpoint_pth', type = str, default = None, help = "Start Checkpoint to be loaded")
    parser.add_argument('-epochs', type = int, default = 5, help = "Start Checkpoint to be loaded")
    parser.add_argument('-eval_step', type = int, default= 100, help="Steps between two different evaluations")
    parser.add_argument('-run_parallel', type = bool, default= False, help="If you want to do parallel training")
    args = parser.parse_args()
    return args


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
    steps = checkpoint['steps']
    logging_step_list = checkpoint['loss_profile']['logging_step_list']
    tr_running_loss = checkpoint['loss_profile']['training_running_loss']
    val_running_loss = checkpoint['loss_profile']['validation_running_loss']
    return steps, logging_step_list, tr_running_loss, val_running_loss


'''
###################################################################################
If "dataset" is serialized then load that else create it and save that
###################################################################################
'''
def load_create_dataset(serialized_path, raw_path, max_seq_len, max_ev_len):
    if os.path.isfile(serialized_path):
        dataset = torch.load(serialized_path)
    else:
        dataset = tree2bert_dataset(data_path = raw_path,
                                    max_seq_len = max_seq_len,
                                    max_ev_len = max_ev_len)
        torch.save(dataset, serialized_path)
    return dataset

'''
#######################################################################################
Test loss on the test loaded using the model - return the loss and perplexity on val set
#######################################################################################
'''
def test(model, te_loader, device):
    USING_PARALLEL = isinstance(model, torch.nn.DataParallel)

    model.eval()
    te_loss = []
    for idx, batch in tqdm(enumerate(te_loader), total=len(te_loader), leave=True, desc = 'Evaluating !!'):
        
        text_tok_src = batch['text_tok_src'].long().to(device)
        text_mask_src = batch['text_mask_src'].long().to(device)
        event_loc_src = batch['event_loc_src'].long().to(device)
        event_mask_src = batch['event_mask_src'].long().to(device)
        text_tok_tar = batch['text_tok_tar'].long().to(device)
        text_mask_tar = batch['text_mask_tar'].long().to(device)

        # In hugginface encoder-decoder model 
        label = cc( text_tok_tar )
        label[label == 0] = -100

        # forward
        decoder_out = model(src_text_tok = text_tok_src, src_text_mask = text_mask_src, 
                            src_event_loc = event_loc_src, src_event_pad_mask = event_mask_src, 
                            tar_text_tok = text_tok_tar, tar_pad_mask = text_mask_tar)

        te_loss.append(decoder_out.loss.item()) if USING_PARALLEL == False else te_loss.append(decoder_out.loss.mean().item())
        if idx >= 10:
            break

    model.train()
    return np.mean(te_loss)


'''
###################################################################################
Defining the main() function
###################################################################################
'''

if __name__ == '__main__':  
    time_start = time.time()
    args = parse()
    raw_root_path = './../Data/raw/seq2seq'
    serialized_root_path = './../Data/serialized/seq2seq'
    
    print(args)
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device being used is {} !!'.format(device))


    '''==== Bert tokenizer - used for encoding the text ===='''
    fast_tokenizer = BertWordPieceTokenizer(vocab='./vocab/vocab.txt')
    fast_tokenizer.add_special_tokens(['[et_sep]', '[ea_sep]', '[ds_sep]'])
    fast_tokenizer.enable_truncation(max_length = args.max_seq_len)
    fast_tokenizer.enable_padding(length = args.max_seq_len)


    val_path = 'train_short_prefix.txt.tst'
    serialized_val_path = os.path.join(serialized_root_path, val_path) + '.pt'
    raw_val_path = os.path.join(raw_root_path, val_path)
    val_dataset = load_create_dataset(serialized_path = serialized_val_path, 
                                     raw_path = raw_val_path, max_seq_len = args.max_seq_len, max_ev_len = args.max_ev_len)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)


    # Pre-train BERTs and then load the weights
    config = Config()

    model = TreeBERT2BERT(config)
    model.to(device)

    # Defining the optimizer and the LR scheudler
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0001},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-7)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.3*args.max_steps), 
                                                num_training_steps = args.max_steps)
    
    steps = 0
    tr_loss = []
    val_loss = []
    tr_running_loss = []
    val_running_loss = []
    logging_step_list = []
    best_val_loss = 1e20

    if args.load_checkpoint_pth != None:
        result = load_checkpoint(model= model, optimizer = optimizer, scheduler = scheduler,
                                steps = steps,  logging_step_list = logging_step_list, 
                                tr_running_loss = tr_running_loss,  val_running_loss = val_running_loss,
                                path = args.load_checkpoint_pth)
        steps, logging_step_list, tr_running_loss, val_running_loss = result
        best_val_loss = val_running_loss[-1]

    # Placing the model in multiple GPUs
    if ( n_gpus > 1 ) and ( args.run_parallel == True ):
        print(f'\n>> No of GPUs available are :: {n_gpus}\n')
        model = torch.nn.DataParallel(model, device_ids = [0, 1]).to(device)
        print('>> RUNNING IN PARALLEL MODE !!')    
    USING_PARALLEL = isinstance(model, torch.nn.DataParallel)

    model.train()

    stop_run = False
    for epochs in range(args.epochs):    
        if stop_run == True:
            break

        for file_idx in range(1, 11):
            file_name = 'train_short_prefix.txt.' + f'{file_idx}'
            serialized_tr_path = os.path.join(serialized_root_path, file_name) + '.pt'
            raw_tr_path = os.path.join(raw_root_path, file_name)
            tr_dataset = load_create_dataset(serialized_path = serialized_tr_path, raw_path = raw_tr_path, 
                                               max_seq_len = args.max_seq_len, max_ev_len= args.max_ev_len)
            
            tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size = args.batch_size, shuffle = True)
            
            if stop_run == True:
                break

            for idx, batch in tqdm(enumerate(tr_loader), total=len(tr_loader), leave=True, desc='Training !!'):

                text_tok_src = batch['text_tok_src'].long().to(device)
                text_mask_src = batch['text_mask_src'].long().to(device)
                event_loc_src = batch['event_loc_src'].long().to(device)
                event_mask_src = batch['event_mask_src'].long().to(device)
                text_tok_tar = batch['text_tok_tar'].long().to(device)
                text_mask_tar = batch['text_mask_tar'].long().to(device)

                decoder_out = model(src_text_tok = text_tok_src, src_text_mask = text_mask_src, 
                                    src_event_loc = event_loc_src, src_event_pad_mask = event_mask_src, 
                                    tar_text_tok = text_tok_tar, tar_pad_mask = text_mask_tar)

                tr_loss.append(decoder_out.loss.item()) if USING_PARALLEL == False else tr_loss.append(decoder_out.loss.mean().item())
                optimizer.zero_grad()
                decoder_out.loss.backward() if USING_PARALLEL == False else decoder_out.loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm = 1.0)
                optimizer.step()
                scheduler.step()

                ### After every eval steps, compute the running average for training and validation loss
                if steps % args.eval_step == 0:
                    del text_tok_src, text_mask_src, text_tok_tar, text_mask_tar
                    tr_loss = np.mean(tr_loss)
                    val_loss = test(model = model, te_loader = val_loader, device = device)   
                    assert model.training == True

                    tr_running_loss.append(tr_loss)
                    val_running_loss.append(val_loss)
                    logging_step_list.append(steps)
                    loss_profile = {'training_running_loss': tr_running_loss,
                                    'validation_running_loss': val_running_loss,
                                    'logging_step_list': logging_step_list}
                    torch.save(loss_profile, './../model_checkpoints/TREEBERT2BERT/loss_profile.pt')

                    print(f'\n>> Loss at epoch {epochs} and step {steps} is :: Tr_loss= {tr_loss}, Val_loss= {val_loss}')
                    tr_loss = []

                    # if val_loss at this step is better than (<) best valid_loss then save the checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ts = time.time()
                        save_checkpoint(model= model, optimizer= optimizer, scheduler= scheduler, 
                                        steps= steps, loss_profile = loss_profile,
                                        path= './../model_checkpoints/TREEBERT2BERT/best_val_model.pt' )

                steps += 1
                if steps == args.max_steps:
                    stop_run = True
                    break

    time_end = time.time()
    print(f'\nTotal time taken :: {time_end-time_start}')