import torch
from torch.utils.data import Dataset
import gc
import numpy as np
from copy import deepcopy as cc
import pdb
from tqdm import tqdm
import argparse
import os
from tokenizers import BertWordPieceTokenizer
import sys
gc.disable()
gc.collect()

def parse():
    parser = argparse.ArgumentParser(description = "Pre-training Tree-BERT with masked lanugage modelling.")
    parser.add_argument('-max_seq_len', type=int, default = 512, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_ev_len', type=int, default = 34, help = "Max token in input sentence that can be processed")
    parser.add_argument('-model_class', type=str, default = 'mlm', help = "model for which dataset is to be generated")
    args = parser.parse_args()
    return args


def find_all_indices(text_tok, element):
    indices = [i for i, x in enumerate(text_tok) if x == element]
    return indices

'''========================================================================================================================
                                            MLM dataset utils
========================================================================================================================'''

def process_tree2bert_mlm(data_src, tokenizer, max_seq_len = 512, max_event_len = 34):
    data_processed_src = tokenizer.encode_batch(data_src)

    et_sep_id = tokenizer.token_to_id('[et_sep]')
    sep_id = tokenizer.token_to_id('[SEP]')
    mask_id = tokenizer.token_to_id('[MASK]')
    final_text_tok_src = []
    final_text_mask_src = []
    final_event_loc_src = []
    final_event_mask_src = []
    final_mlm_ip = []
    final_mlm_op = []

    for sentence_src in tqdm(data_processed_src):
        src_space_left = max_seq_len - sum(sentence_src.attention_mask)

        src_event_loc = find_all_indices(sentence_src.ids, et_sep_id)    # find indices of [et_sep] tokens
        src_space_to_be_filled = max_event_len - len(src_event_loc)      # no. of tokens required for additional/dummy [et_sep] = T

        src_event_mask = [1]*(len(src_event_loc)) + [0]*src_space_to_be_filled   # mask of the event_locations
        src_sep_idx = sentence_src.ids.index(sep_id) + 1                 # location of [SEP]    
        src_event_loc +=  list(range(src_sep_idx, (src_sep_idx + src_space_to_be_filled)))  # idx of next T tokens (after [SEP]) is the idx of dummy [et_sep] 

        if (src_space_left > 0) and (src_space_left >= src_space_to_be_filled) and (src_space_to_be_filled >= 0):

            text_tok = torch.tensor(sentence_src.ids).reshape(1, -1)
            text_mask = torch.tensor(sentence_src.attention_mask).reshape(1, -1)
            event_loc = torch.tensor(src_event_loc).reshape(1, -1)
            event_mask = torch.tensor(src_event_mask).reshape(1, -1)

            #--------------- CREATING THE MLM DATASET-------------------------#

            SENTENCE_LEN = text_mask.sum().item()
            text_tok_subset = text_tok[:,:SENTENCE_LEN]

            # mlm_target = torch.ones(size = (SENTENCE_LEN, ))*(-100)
            mlm_ip = cc(text_tok)
            et_sep_indices = event_loc[event_mask.bool()]

            coin_tosses = torch.rand(size = (SENTENCE_LEN,))               # toss S coins 
            coin_tosses[0], coin_tosses[-1] = 0.0, 0.0                     # set the 1st and the last index to 0.0
            coin_tosses = torch.scatter(input = coin_tosses, dim = 0,      # Set all the [et_sep] indices to 0.0
                                        index = et_sep_indices, src = torch.zeros_like(et_sep_indices).float())    

            mlm_target = torch.where(coin_tosses >= torch.tensor(0.85), 
                                        text_tok_subset, -100)     # if coin toss > 0.85, mlm_target = token_index else it is -100

            mlm_ip = torch.where(coin_tosses >= torch.tensor(0.985),        # if coin toss > 0.985 (.15*.1), mlm_ip = RANDOM else it is "token index"
                                torch.randint(low = 1, high = 30520, size =text_tok_subset.shape)[0], 
                                text_tok_subset)

            mlm_ip = torch.where(torch.logical_and(coin_tosses >= torch.tensor(0.865), coin_tosses < torch.tensor(0.985)), 
                                    mask_id,                                # if coin toss in [.865, .985], mlm_ip = [MASK] else "token index"  
                                    mlm_ip)      

            mlm_target = torch.cat( (mlm_target, -100 * torch.ones_like(text_tok[:, SENTENCE_LEN:])), dim = 1)
            mlm_ip = torch.cat( (mlm_ip, text_tok[:, SENTENCE_LEN:]), dim = 1)
            #--------------- CREATING THE MLM DATASET-------------------------#

            final_text_tok_src.append(text_tok)
            final_text_mask_src.append(text_mask)
            final_event_loc_src.append(event_loc)
            final_event_mask_src.append(event_mask)
            final_mlm_ip.append(mlm_ip)
            final_mlm_op.append(mlm_target)

    return {'src_text_tok': torch.cat(final_text_tok_src).type(torch.int32), 'src_text_mask': torch.cat(final_text_mask_src).type(torch.int32),
           'src_event_loc': torch.cat(final_event_loc_src).type(torch.int32), 'src_event_mask': torch.cat(final_event_mask_src).type(torch.int32),
           'mlm_ip': torch.cat(final_mlm_ip).type(torch.int32), 'mlm_op': torch.cat(final_mlm_op).type(torch.int32)}

'''========================================================================================================================'''
def get_treeBERT_mlm_dataset(args, data_path, tokenizer):

    sentence_list = open(os.path.join('Data/raw/seq2seq', data_path)).read().splitlines()      
    for i in range(3, 5):
        sentence_list_src = []
        out_path = os.path.join('Data/serialized/seq2seq/mlm', data_path) + '.' + str(i) + '.pt'    
        for sentence in  sentence_list:
            src = sentence.strip()
            src = " ".join( src.split('[ds_sep]'))        # replace [ds_sep] with " " 
            sentence_list_src.append(src)

        print(f'>> Processing :: {out_path}')
        result = process_tree2bert_mlm(data_src=sentence_list_src, tokenizer = tokenizer, 
                                            max_seq_len = args.max_seq_len, max_event_len = args.max_ev_len)
        torch.save(obj = result, f = out_path)
    return

'''========================================================================================================================'''
class tree2bert_dataset_mlm(Dataset):

    def __init__(self, data_path, max_seq_len = 512, max_ev_len = 34):
        self.data = torch.load(data_path)

    def __len__(self):
        key = list(self.data.keys())[0]
        return len(self.data[key])

    def __getitem__(self, index) :
        return {'text_tok_src': self.data[index]['src_text_tok'], 'text_mask_src': self.data[index]['src_text_mask'],
                'event_loc_src': self.data[index]['src_event_loc'], 'event_mask_src': self.data[index]['src_event_mask'],
                'mlm_input': self.data[index]['mlm_ip'], 'mlm_target': self.data[index]['mlm_op']}

                 
'''========================================================================================================================
                                            Seq 2 Seq dataset utils
========================================================================================================================'''
def process_tree2bert(data_src, data_tar, tokenizer, max_seq_len = 512, max_event_len = 34):
    data_processed_src = tokenizer.encode_batch(data_src)
    data_processed_tar = tokenizer.encode_batch(data_tar)

    et_sep_id = tokenizer.token_to_id('[et_sep]')
    sep_id = tokenizer.token_to_id('[SEP]')
    final_text_tok_src = []
    final_text_mask_src = []
    final_event_loc_src = []
    final_event_mask_src = []
    final_text_tok_tar = []
    final_text_mask_tar = []


    for sentence_src, sentence_tar in tqdm(zip(data_processed_src, data_processed_tar), total=len(data_processed_tar)):
    # for sentence_src, sentence_tar in zip(data_processed_src, data_processed_tar):

        src_space_left = max_seq_len - sum(sentence_src.attention_mask)

        src_event_loc = find_all_indices(sentence_src.ids, et_sep_id)    # find indices of [et_sep] tokens
        src_space_to_be_filled = max_event_len - len(src_event_loc)      # no. of tokens required for additional/dummy [et_sep] = T

        src_event_mask = [1]*(len(src_event_loc)) + [0]*src_space_to_be_filled   # mask of the event_locations
        src_sep_idx = sentence_src.ids.index(sep_id) + 1                 # location of [SEP]    
        src_event_loc +=  list(range(src_sep_idx, (src_sep_idx + src_space_to_be_filled)))  # idx of next T tokens (after [SEP]) is the idx of dummy [et_sep] 

        if (src_space_left > 0) and (src_space_left >= src_space_to_be_filled) and (src_space_to_be_filled >= 0):

            text_tok = torch.tensor(sentence_src.ids).reshape(1, -1)
            text_mask = torch.tensor(sentence_src.attention_mask).reshape(1, -1)
            event_loc = torch.tensor(src_event_loc).reshape(1, -1)
            event_mask = torch.tensor(src_event_mask).reshape(1, -1)

            text_tok_tar = torch.tensor(sentence_tar.ids).reshape(1, -1)
            text_mask_tar = torch.tensor(sentence_tar.attention_mask).reshape(1, -1)

            final_text_tok_src.append(text_tok)
            final_text_mask_src.append(text_mask)
            final_event_loc_src.append(event_loc)
            final_event_mask_src.append(event_mask)
            final_text_tok_tar.append(text_tok_tar)
            final_text_mask_tar.append(text_mask_tar)
    
    return {'src_text_tok': torch.cat(final_text_tok_src).type(torch.int32), 'src_text_mask': torch.cat(final_text_mask_src).type(torch.int32),
            'src_event_loc': torch.cat(final_event_loc_src).type(torch.int32), 'src_event_mask': torch.cat(final_event_mask_src).type(torch.int32),
            'tar_text_tok': torch.cat(final_text_tok_tar).type(torch.int32), 'tar_text_mask': torch.cat(final_text_mask_tar).type(torch.int32)}

'''========================================================================================================================'''
def get_treeBERT_dataset(data_path, tokenizer, max_seq_len, max_ev_len, out_path):

    sentence_list = open(os.path.join(data_path)).read().splitlines()      

    sentence_list_src = []
    sentence_list_tar = []

    for idx, sentence in enumerate(sentence_list):
        try:
            src, tar = sentence.split('[ds_sep]')
            sentence_list_src.append(src.strip())
            sentence_list_tar.append(tar.strip())

        except ValueError as e:
            print("[DS_SEP] errors!")
            print(f'Line # :: {idx}, \n {sentence}')
            # sys.exit("[DS_SEP] errors!")

    print(f'>> Processing :: {out_path}')
    result = process_tree2bert(data_src=sentence_list_src, data_tar = sentence_list_tar, tokenizer = tokenizer, 
                                            max_seq_len = max_seq_len, max_event_len = max_ev_len)
    torch.save(obj = result, f = out_path)
    return


'''========================================================================================================================'''
class tree2bert_dataset(Dataset):

    def __init__(self, data_path, max_seq_len = 512, max_ev_len = 34):
        self.data = torch.load(data_path)

    def __len__(self):
        key = list(self.data.keys())[0]
        return len(self.data[key])

    def __getitem__(self, index) :
        return {'text_tok_src': self.data['src_text_tok'][index], 'text_mask_src': self.data['src_text_mask'][index],
                'event_loc_src': self.data['src_event_loc'][index], 'event_mask_src': self.data['src_event_mask'][index],
                'text_tok_tar': self.data['tar_text_tok'][index], 'text_mask_tar': self.data['tar_text_mask'][index]}


'''========================================================================================================================'''
'''========================================================================================================================'''
class test_dataset(Dataset):
    def __init__(self, data_path, max_seq_len, max_ev_len):

        sentence_list = open(data_path).read().splitlines()
        sentence_list_src = []
        
        self.fast_tokenizer = BertWordPieceTokenizer(vocab='./vocab/vocab.txt')
        self.fast_tokenizer.add_special_tokens(['[et_sep]', '[ea_sep]', '[ds_sep]'])
        self.fast_tokenizer.enable_truncation(max_length = max_seq_len)
        self.fast_tokenizer.enable_padding(length=max_seq_len)
        
        no_of_options = len(sentence_list[0].split('[ds_sep]')) - 1
        self.sentence_list_tar = [[] for i in range(no_of_options)]
        
        for sentence in  sentence_list:
            event_sequences = sentence.strip().split('[ds_sep]')
            src = event_sequences[0]
            tar_list = event_sequences[1:]
            src = src.strip() if src.strip().endswith('[et_sep]') else (src.strip() + ' [et_sep]')
            tar_list = [tar.strip() if tar.strip().endswith('[et_sep]') else (tar.strip() + ' [et_sep]') for tar in tar_list]
            sentence_list_src.append(src)
            for j in range(no_of_options):
                self.sentence_list_tar[j].append(tar_list[j])

        self.result = []
        for sentence_tar in self.sentence_list_tar:
            result_temp = process_tree2bert(data_src=sentence_list_src, data_tar=sentence_tar, 
                                   tokenizer = self.fast_tokenizer, max_seq_len=max_seq_len,
                                   max_event_len=max_ev_len) 
            self.result.append(result_temp)

        del sentence_list_src
    def __len__(self):
        return self.result[0]['src_text_tok'].shape[0]

    def __getitem__(self, idx):
        return {'text_tok_src': self.result[0]['src_text_tok'][idx], 'text_mask_src': self.result[0]['src_text_mask'][idx],
                'event_loc_src': self.result[0]['src_event_loc'][idx], 'event_mask_src': self.result[0]['src_event_mask'][idx],
                'text_tok_tar0': self.result[0]['tar_text_tok'][idx], 'text_mask_tar0': self.result[0]['tar_text_mask'][idx],
                'text_tok_tar1': self.result[1]['tar_text_tok'][idx], 'text_mask_tar1': self.result[1]['tar_text_mask'][idx],
                'text_tok_tar2': self.result[2]['tar_text_tok'][idx], 'text_mask_tar2': self.result[2]['tar_text_mask'][idx],
                'text_tok_tar3': self.result[3]['tar_text_tok'][idx], 'text_mask_tar3': self.result[3]['tar_text_mask'][idx],
                'text_tok_tar4': self.result[4]['tar_text_tok'][idx], 'text_mask_tar4': self.result[4]['tar_text_mask'][idx],
                'text_tok_tar5': self.result[5]['tar_text_tok'][idx], 'text_mask_tar5': self.result[5]['tar_text_mask'][idx]}


'''========================================================================================================================'''
'''========================================================================================================================'''
'''========================================================================================================================'''
if __name__ == '__main__':
    args = parse()

    src_folder = 'Data/raw/seq2seq'

    fast_tokenizer = BertWordPieceTokenizer(vocab='./vocab/vocab.txt')
    fast_tokenizer.add_special_tokens(['[et_sep]', '[ea_sep]', '[ds_sep]'])
    fast_tokenizer.enable_truncation(max_length = args.max_seq_len)
    fast_tokenizer.enable_padding(length = args.max_seq_len)

    # data_paths = ['train_long_prefix.txt.3', 'train_long_prefix.txt.2', 'train_long_prefix.txt.1', 'train_long_prefix.txt.val']
    data_paths = ['train_new11.txt', 'train_new12.txt', 'train_new21.txt', 'train_new22.txt', 'val_new.txt']

    if args.model_class == 'mlm':
        for data_path in data_paths:
            get_treeBERT_mlm_dataset(args, data_path, fast_tokenizer)

    else:
        tar_folder = 'Data/serialized/seq2seq'
        
        for data_path in data_paths:
            in_file_name = data_path             
            out_file_name = data_path + '.pt'

            data_path = os.path.join(src_folder, in_file_name)
            out_path = os.path.join(tar_folder, out_file_name)

            print(f'>> File being processed :: {out_path}')

            get_treeBERT_dataset(data_path, tokenizer = fast_tokenizer, max_seq_len = 512, max_ev_len = 34, out_path = out_path)