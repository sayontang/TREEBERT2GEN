import torch
from transformers import  BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from copy import deepcopy as cc
from tqdm import tqdm
import time
import pdb
import os
import numpy as np
from TreeBERT import *
from torch import nn
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


'''
###################################################################################
Defining the argument parser
###################################################################################
'''
def parse():
    parser = argparse.ArgumentParser(description = "Pre-training Tree-BERT with masked lanugage modelling.")
    parser.add_argument('-max_seq_len', type=int, default = 512, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_ev_len', type=int, default = 34, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_steps', type = int, default= 300000, help="Number of epochs")
    parser.add_argument('-batch_size', type = int, default= 20, help="Batch size")
    parser.add_argument('-load_checkpoint_pth', type = str, default = None, help = "Start Checkpoint to be loaded")
    parser.add_argument('-epochs', type = int, default = 9, help = "Start Checkpoint to be loaded")
    parser.add_argument('-eval_step', type = int, default= 2500, help="Steps between two different evaluations")
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
Defination of the dataloader for mlm
###################################################################################
'''
class tree2bert_dataset_mlm(Dataset):

    def __init__(self, data_path, max_seq_len = 512, max_ev_len = 34):
        self.data = torch.load(data_path)

    def __len__(self):
        key = list(self.data.keys())[0]
        return len(self.data[key])

    def __getitem__(self, index) :
        return {'text_tok_src': self.data['src_text_tok'][index], 'text_mask_src': self.data['src_text_mask'][index],
                'event_loc_src': self.data['src_event_loc'][index], 'event_mask_src': self.data['src_event_mask'][index],
                'mlm_input': self.data['mlm_ip'][index], 'mlm_target': self.data['mlm_op'][index]}

'''
#######################################################################################
Test loss on the test loaded using the model - return the loss and perplexity on val set
#######################################################################################
'''
def test(model, te_loader, device):
    USING_PARALLEL = isinstance(model, torch.nn.DataParallel)

    model.eval()
    te_loss = []
    model_pred = []
    true_val = []

    # for idx, batch in tqdm(enumerate(te_loader), total=len(te_loader), leave=True, desc = 'Evaluating !!'):
    for idx, batch in enumerate(te_loader):
        
        #==============================================================================================#
        '''                Dynamic batch size --> creating the required variables'''
        #==============================================================================================#
        text_mask_src = batch['text_mask_src'].long().to(device)
        mlm_input = batch['mlm_input'].long().to(device)
        mlm_target = batch['mlm_target'].long().to(device)

        max_tok_len_in_batch = text_mask_src.sum(1).max().item()
        text_mask_src = text_mask_src[:, :max_tok_len_in_batch]
        mlm_input = mlm_input[:, :max_tok_len_in_batch]
        mlm_target = mlm_target[:, :max_tok_len_in_batch].reshape(-1)
        #==============================================================================================#
        '''---------------xx Dynamic batch size --> creating the required variables xx---------------'''
        #==============================================================================================#

        # forward
        decoder_out = model(input_ids = mlm_input, attention_mask = text_mask_src).logits
        decoder_out = decoder_out.reshape(-1, 30522)    # (BxS) x V
        
        model_pred.extend(torch.argmax(decoder_out[mlm_target != -100, :], dim = -1).cpu().reshape(-1).numpy() )
        true_val.extend(mlm_target[ mlm_target!= -100].cpu().numpy())

        decoder_out_loss = CE_loss(input = decoder_out, target = mlm_target) 
        if USING_PARALLEL == True:
            decoder_out_loss = decoder_out_loss.mean()
            
        te_loss.append(decoder_out_loss.item())

    val_acc = sum(np.array(model_pred) == np.array(true_val) )/len(true_val)
    model.train()
    return np.mean(te_loss), val_acc


'''
###################################################################################
Defining the main() function
###################################################################################
'''

if __name__ == '__main__':  
    time_start = time.time()
    
    args = parse()
    raw_root_path = './Data/raw/seq2seq'
    serialized_root_path = './Data/serialized/seq2seq/mlm'
    
    print(args)
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device being used is {} !!'.format(device))

    val_path = 'train_long_prefix.txt.val.1.pt'
    serialized_val_path = os.path.join(serialized_root_path, val_path)
    raw_val_path = os.path.join(raw_root_path, val_path)
    val_dataset = tree2bert_dataset_mlm(data_path = serialized_val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True)

    # Pre-train BERTs and then load the weights
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)

    # Defining the optimizer and the LR scheudler
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0001},
                                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                    'weight_decay_rate': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-7)
    # scheduler = ReduceLROnPlateau(optimizer, factor = 0.7, patience = 4)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.3*args.max_steps), 
                                                num_training_steps = args.max_steps)

    CE_loss = torch.nn.CrossEntropyLoss(ignore_index = -100)
    
    steps = 0
    tr_loss = []
    val_loss = []
    tr_running_loss = []
    val_running_loss = []
    val_running_acc = []
    logging_step_list = []
    best_val_loss = 1e20
    stop_run = False
    ip_data_files = ['train_long_prefix.txt.1.4.pt', 'train_long_prefix.txt.2.4.pt', 'train_long_prefix.txt.3.3.pt',
                     'train_long_prefix.txt.1.3.pt', 'train_long_prefix.txt.2.1.pt', 'train_long_prefix.txt.3.1.pt',
                     'train_long_prefix.txt.1.2.pt', 'train_long_prefix.txt.2.2.pt', 'train_long_prefix.txt.3.4.pt',
                     'train_long_prefix.txt.1.1.pt', 'train_long_prefix.txt.2.3.pt', 'train_long_prefix.txt.3.2.pt']    

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

    
    for file_name in ip_data_files:
        if stop_run == True:
            break

        serialized_tr_path = os.path.join(serialized_root_path, file_name)
        tr_dataset = tree2bert_dataset_mlm(data_path = serialized_tr_path)
        tr_loader = torch.utils.data.DataLoader(tr_dataset, batch_size = args.batch_size, shuffle = True)


        for idx, batch in enumerate(tr_loader):
        # for idx, batch in tqdm(enumerate(tr_loader), total=len(tr_loader), leave=True, desc='Training !!'):

            #==============================================================================================#
            '''                Dynamic batch size --> creating the required variables'''
            #==============================================================================================#
            text_mask_src = batch['text_mask_src'].long().to(device)
            mlm_input = batch['mlm_input'].long().to(device)
            mlm_target = batch['mlm_target'].long().to(device)
            
            max_tok_len_in_batch = text_mask_src.sum(1).max().item()
            text_mask_src = text_mask_src[:, :max_tok_len_in_batch]
            mlm_input = mlm_input[:, :max_tok_len_in_batch]
            mlm_target = mlm_target[:, :max_tok_len_in_batch].reshape(-1)
            #==============================================================================================#
            '''---------------xx Dynamic batch size --> creating the required variables xx---------------'''
            #==============================================================================================#

            decoder_out = model(input_ids = mlm_input, attention_mask = text_mask_src)
            decoder_out = decoder_out.logits
            decoder_out_loss = CE_loss(input = decoder_out.reshape(-1, 30522), target = mlm_target.reshape(-1)) 
            if USING_PARALLEL == True:
                decoder_out_loss = decoder_out_loss.mean()

            tr_loss.append(decoder_out_loss.item())
            optimizer.zero_grad()
            decoder_out_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm = 1.0)
            optimizer.step()
            scheduler.step()

            ### After every eval steps, compute the running average for training and validation loss
            if steps % args.eval_step == 0:

                tr_loss = np.mean(tr_loss)
                val_loss, val_acc = test(model = model, te_loader = val_loader, device = device)   
                assert model.training == True

                tr_running_loss.append(tr_loss)
                val_running_loss.append(val_loss)
                val_running_acc.append(val_acc)
                
                # Update the learning rate
                # scheduler.step(val_loss)

                # Log the losses by saving it into loss_profile
                logging_step_list.append(steps)
                loss_profile = {'training_running_loss': tr_running_loss,
                                'validation_running_loss': val_running_loss,
                                'validation_running_accuracy': val_running_acc,
                                'logging_step_list': logging_step_list}
                torch.save(loss_profile, './model_checkpoints/BERT_MLM/loss_profile_300k.pt')

                print(f'\n>> Loss at step {steps} is :: Tr_loss= {tr_loss}, Val_loss= {val_loss}, Val_acc = {val_acc}, Best_val_loss= {best_val_loss}')
                tr_loss = []

                # if val_loss at this step is better than (<) best valid_loss then save the checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ts = time.time()
                    save_checkpoint(model= model, optimizer= optimizer, scheduler= scheduler, 
                                    steps= steps, loss_profile = loss_profile,
                                    path= './model_checkpoints/BERT_MLM/best_val_model_300k.pt' )

            steps += 1
            if steps == args.max_steps:
                stop_run = True
                break