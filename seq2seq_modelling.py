import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
import time
import pdb
import os
import numpy as np
from event_dataset import *
from utils import *
from models import *

'''
###################################################################################
Defining the argument parser
###################################################################################
'''
def parse():
    parser = argparse.ArgumentParser(description = "Conditional event sequence generation with Tree transformer as encoder and BERT as decoder.")
    parser.add_argument('-max_seq_len', type=int, default = 512, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_ev_len', type=int, default = 34, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_steps', type = int, default= 200000, help="Number of epochs")
    parser.add_argument('-batch_size', type = int, default= 20, help="Batch size")
    parser.add_argument('-load_checkpoint_pth', type = str, default = None, help = "Start Checkpoint to be loaded")
    parser.add_argument('-epochs', type = int, default = 3, help = "Start Checkpoint to be loaded")
    parser.add_argument('-eval_step', type = int, default= 2000, help="Steps between two different evaluations")
    parser.add_argument('-run_parallel', type = bool, default= False, help="If you want to do parallel training")
    parser.add_argument('-enc_loading_checkpoint', type=str, help = "checkpoint for loading the encoder")
    parser.add_argument('-dec_loading_checkpoint', type=str, help = "checkpoint for loading the decoder")
    parser.add_argument('-model_chkpt', type=str, help = "Checkpoint where the model is to be saved")
    parser.add_argument('-et_sep', type = str, default = 'False', help = "set et_sep = True if you just want to pass the [et_sep] representation from the encoder as input the the decoder")
    parser.add_argument('-model_type', type = str, default = 'treebert2bert', help = "Either bert2bert or treebert2bert")    
    args = parser.parse_args()
    return args

'''
#######################################################################################
Test loss on the test loaded using the model - return the loss and perplexity on val set
#######################################################################################
'''

def test(model, te_loader, device):
    USING_PARALLEL = isinstance(model, torch.nn.DataParallel)

    model.eval()
    te_loss = []
    # for idx, batch in tqdm(enumerate(te_loader), total=len(te_loader), leave=True, desc = 'Evaluating !!'):
    for idx, batch in enumerate(te_loader):
        text_tok_src, text_mask_src, event_loc_src, event_mask_src, text_tok_tar, text_mask_tar = dynamic_batch(batch=batch, device=device)
        decoder_out = model(src_text_tok = text_tok_src, src_text_mask = text_mask_src, 
                            src_event_loc = event_loc_src, src_event_pad_mask = event_mask_src, 
                            tar_text_tok = text_tok_tar, tar_pad_mask = text_mask_tar)

        te_loss.append(decoder_out.loss.item()) if USING_PARALLEL == False else te_loss.append(decoder_out.loss.mean().item())
        
    model.train()
    return np.mean(te_loss)

'''
###################################################################################
Defining the main() function
###################################################################################
'''

if __name__ == '__main__':  
    torch.manual_seed(0)
    time_start = time.time()
    args = parse()
    args.et_sep = args.et_sep == 'True'
    print(f'Arguments are :: {args}')
    raw_root_path = './Data/raw/seq2seq'
    serialized_root_path = './Data/serialized/seq2seq'

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device being used is {} !!'.format(device))
    
    # val_path = 'train_long_prefix.txt.val.1_seq2seq.pt'
    val_path = 'val_new.txt.pt'
    serialized_val_path = os.path.join(serialized_root_path, val_path)
    val_dataset = tree2bert_dataset(data_path= serialized_val_path, max_seq_len=512, max_ev_len=34)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)

    # Define the model and load the weights it loading from checkpoint
    if args.model_type == 'bert2bert':
        model = BERT2BERT(args)

    if args.model_type == 'treebert2bert':
        config = Config()
        model = TreeBERT2BERT(config, args)

    if args.load_checkpoint_pth != None:
        checkpoint = torch.load(f = args.load_checkpoint_pth, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        print(f'>> Finished loading checkpoint from {args.load_checkpoint_pth}')        

    model.to(device)

    # Defining the optimizer and the LR scheudler
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0001},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-7)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.3*args.max_steps), 
                                                num_training_steps = args.max_steps)
    # scheduler = ReduceLROnPlateau(optimizer, factor = 0.7, patience = 4)

    steps = 0
    tr_loss = []
    val_loss = []
    tr_running_loss = []
    val_running_loss = []
    logging_step_list = []
    best_val_loss = 1e20
    stop_run = False

    # ip_data_files = ['train_long_prefix.txt.1.1_seq2seq.pt', 'train_long_prefix.txt.2.1_seq2seq.pt', 'train_long_prefix.txt.3.1_seq2seq.pt']
    ip_data_files = ['train_new11.txt.pt', 'train_new12.txt.pt', 'train_new21.txt.pt', 'train_new22.txt.pt']

    # Placing the model in multiple GPUs
    if ( n_gpus > 1 ) and ( args.run_parallel == True ):
        print(f'\n>> No of GPUs available are :: {n_gpus}\n')
        model = torch.nn.DataParallel(model, device_ids = [0, 1]).to(device)
        print('>> RUNNING IN PARALLEL MODE !!')    
    USING_PARALLEL = isinstance(model, torch.nn.DataParallel)

    model.train()

    for epoch in range(args.epochs):
        if stop_run == True:
            break

        for file_name in ip_data_files:
            if stop_run == True:
                break
            
            serialized_tr_path = os.path.join(serialized_root_path, file_name) 
            tr_dataset = tree2bert_dataset(data_path= serialized_tr_path, max_seq_len=512, max_ev_len=34)
            tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size = args.batch_size, shuffle = True)

            # for idx, batch in tqdm(enumerate(tr_loader), total=len(tr_loader), leave=True, desc='Training !!'):
            for idx, batch in enumerate(tr_loader):

                text_tok_src, text_mask_src, event_loc_src, event_mask_src, text_tok_tar, text_mask_tar = dynamic_batch(batch=batch, device=device)
                
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

                    torch.save(loss_profile, os.path.join('./model_checkpoints', ('loss_' + args.model_chkpt) ) + '.pt')

                    print(f'\n>> Loss at step {steps} is :: Tr_loss= {tr_loss}, Val_loss= {val_loss}')
                    tr_loss = []

                    # if val_loss at this step is better than (>) best valid_loss then save the checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        ts = time.time()
                        save_checkpoint(model= model, optimizer= optimizer, scheduler= scheduler, 
                                        steps= steps, loss_profile = loss_profile,
                                        path = os.path.join('./model_checkpoints', args.model_chkpt) + '.pt')
                        print('>> SAVING MODEL\n')
                steps += 1
                if steps == args.max_steps:
                    stop_run = True
                    break

    time_end = time.time()
    print(f'Total time taken :: {time_end - time_start} secs')