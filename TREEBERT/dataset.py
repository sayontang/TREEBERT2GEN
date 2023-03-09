import torch
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
import pdb

def find_all_indices(text_tok, element):
    indices = [i for i, x in enumerate(text_tok) if x == element]
    return indices


'''=============================================================================================='''
'''=============================================================================================='''

def process_tree2bert(data_src, data_tar, tokenizer, max_seq_len = 512, max_event_len = 30):
    data_processed_src = tokenizer.encode_batch(data_src)
    data_processed_tar = tokenizer.encode_batch(data_tar)

    et_sep_id = tokenizer.token_to_id('[et_sep]')
    pad_id = tokenizer.token_to_id('[PAD]')
    sep_id = tokenizer.token_to_id('[SEP]')

    final_text_tok_src = []
    final_text_mask_src = []
    final_event_loc_src = []
    final_event_mask_src = []

    final_text_tok_tar = []
    final_text_mask_tar = []

    for sentence_src, sentence_tar in zip(data_processed_src, data_processed_tar):
        src_space_left = max_seq_len - sum(sentence_src.attention_mask)

        src_event_loc = find_all_indices(sentence_src.ids, et_sep_id)
        src_space_to_be_filled = max_event_len - len(src_event_loc)

        src_event_mask = [1]*(len(src_event_loc)) + [0]*src_space_to_be_filled
        # src_event_loc +=  find_all_indices(sentence_src.attention_mask, 0)[:src_space_to_be_filled]
        src_sep_idx = sentence_src.ids.index(sep_id) + 1
        src_event_loc +=  list(range(src_sep_idx, (src_sep_idx + src_space_to_be_filled)))

        sentence_src.attention_mask.index(0)
        # as of now only take sequences where no_toks(sentence) <= 512
        # as of now only take sequences where no_events(sentence) <= 30        
        if (src_space_left > 0) and (src_space_left >= src_space_to_be_filled) and (src_space_to_be_filled >= 0):

            final_text_tok_src.append(torch.tensor(sentence_src.ids, dtype = torch.int16).reshape(1, -1))
            final_text_mask_src.append(torch.tensor(sentence_src.attention_mask, dtype = torch.uint8).reshape(1, -1))
            final_event_loc_src.append(torch.tensor(src_event_loc, dtype = torch.int16).reshape(1, -1))
            final_event_mask_src.append(torch.tensor(src_event_mask, dtype = torch.uint8).reshape(1, -1))

            final_text_tok_tar.append(torch.tensor(sentence_tar.ids, dtype = torch.int16).reshape(1, -1))
            final_text_mask_tar.append(torch.tensor(sentence_tar.attention_mask, dtype = torch.uint8).reshape(1, -1))

    return {'src_text_tok': torch.cat(final_text_tok_src).int(), 'src_text_mask': torch.cat(final_text_mask_src),
           'src_event_loc': torch.cat(final_event_loc_src), 'src_event_mask': torch.cat(final_event_mask_src),
           'tar_text_tok': torch.cat(final_text_tok_tar), 'tar_text_mask': torch.cat(final_text_mask_tar)}

'''=============================================================================================='''
class tree2bert_dataset(Dataset):
    def __init__(self, data_path, max_seq_len, max_ev_len):
        
        sentence_list_src = []
        sentence_list_tar = []
        self.fast_tokenizer = BertWordPieceTokenizer(vocab='./vocab/vocab.txt')
        self.fast_tokenizer.add_special_tokens(['[et_sep]', '[ea_sep]', '[ds_sep]'])
        self.fast_tokenizer.enable_truncation(max_length = max_seq_len)
        self.fast_tokenizer.enable_padding(length=max_seq_len)

        sentence_list = open(data_path).read().splitlines()      
        # with open(data_path) as myfile:
        #     sentence_list = [next(myfile) for _ in range(1000)] 

        for sentence in  sentence_list:
            src, tar = sentence.strip().split('[ds_sep]')[:2]
            src = src.strip() if src.strip().endswith('[et_sep]') else (src.strip() + ' [et_sep]')
            tar = tar.strip() if tar.strip().endswith('[et_sep]') else (tar.strip() + ' [et_sep]')
            sentence_list_src.append(src)
            sentence_list_tar.append(tar)

        self.result = process_tree2bert(data_src=sentence_list_src, data_tar=sentence_list_tar, 
                                   tokenizer = self.fast_tokenizer, max_seq_len=max_seq_len,
                                   max_event_len=max_ev_len)
        del sentence_list_src, sentence_list_tar
    def __len__(self):
        return self.result['src_text_tok'].shape[0]

    def __getitem__(self, idx):
        return {'text_tok_src': self.result['src_text_tok'][idx], 'text_mask_src': self.result['src_text_mask'][idx],
                'event_loc_src': self.result['src_event_loc'][idx], 'event_mask_src': self.result['src_event_mask'][idx],
                'text_tok_tar': self.result['tar_text_tok'][idx], 'text_mask_tar': self.result['tar_text_mask'][idx]}
'''=============================================================================================='''
'''=============================================================================================='''


class tree2bert_dataset_test(Dataset):
    def __init__(self, data_path, max_seq_len, max_ev_len):
        
        sentence_list_src = []
        
        self.fast_tokenizer = BertWordPieceTokenizer(vocab='./vocab/vocab.txt')
        self.fast_tokenizer.add_special_tokens(['[et_sep]', '[ea_sep]', '[ds_sep]'])
        self.fast_tokenizer.enable_truncation(max_length = max_seq_len)
        self.fast_tokenizer.enable_padding(length=max_seq_len)
        sentence_list = open(data_path).read().splitlines()
        # with open(data_path) as myfile:
        #     sentence_list = [next(myfile) for x in range(1000)]

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
            result = process_tree2bert(data_src=sentence_list_src, data_tar=sentence_tar, 
                                   tokenizer = self.fast_tokenizer, max_seq_len=max_seq_len,
                                   max_event_len=max_ev_len) 
            self.result.append(result)

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







