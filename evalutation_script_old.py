import torch
from torch.utils.data import  DataLoader
import gc
import numpy as np
from copy import deepcopy as cc
from event_dataset import *
from models import *
from utils import *
from tqdm import tqdm
gc.disable()
gc.collect()
import pdb

def parse():
    parser = argparse.ArgumentParser(description = "Evaluation for conditional event sequence generation.")
    parser.add_argument('-max_seq_len', type=int, default = 256, help = "Max token in input sentence that can be processed")
    parser.add_argument('-max_ev_len', type=int, default = 20, help = "Max token in input sentence that can be processed")
    parser.add_argument('-batch_size', type = int, default= 1, help="Batch size")
    parser.add_argument('-eval_data_path', type = str, help="Path to the evaluation data.")
    parser.add_argument('-load_checkpoint_pth', type=str, help = "Model Checkpoint path")
    parser.add_argument('-et_sep', type = str, default = 'False', help = "set et_sep = True if you just want to pass the [et_sep] representation from the encoder as input the the decoder")
    parser.add_argument('-enc_loading_checkpoint', type=str, help = "Encoder Checkpoint name")
    parser.add_argument('-dec_loading_checkpoint', type=str, help = "Decoder Checkpoint name")
    parser.add_argument('-model_type', type = str, default = 'treebert2bert', help = "Either bert2bert or treebert2bert")    
    args = parser.parse_args()
    return args

if __name__ == '__main__':  

    args = parse()
    print(f'Arguments are :: {args}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device being used is {} !!'.format(device))

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

    model.eval()
    for data_path in os.listdir(args.eval_data_path):
      
        final_answers = np.array([])
        final_generated_sequence = []

        test_ds = test_dataset(data_path = os.path.join(args.eval_data_path, data_path), max_seq_len = 512, max_ev_len = 34)
        test_dl = DataLoader(test_ds, batch_size=1)

        # for idx, batch in tqdm(enumerate(test_dl), total=len(test_dl), leave=True, desc='Training !!'):
        for idx, batch in enumerate(test_dl):

            text_tok_src = batch['text_tok_src'].long().to(device)
            text_mask_src = batch['text_mask_src'].long().to(device)
            event_loc_src = batch['event_loc_src'].long().to(device)
            event_mask_src = batch['event_mask_src'].long().to(device)

            #==================================================================================#
            #                            DYNAMIC BATCHING - ENCODER                            #
            #==================================================================================#
            max_events_in_batch = event_mask_src.sum(1).max().item() + 3
            event_loc_src =event_loc_src[:, :max_events_in_batch]
            event_mask_src = event_mask_src[:, :max_events_in_batch]

            max_tok_len_in_batch = event_loc_src[:, -1].max().item() + 1
            text_tok_src = text_tok_src[:, :max_tok_len_in_batch]
            text_mask_src = text_mask_src[:, :max_tok_len_in_batch]

            answer_per_line = np.array([])
            lines_to_be_written  = []

            for option in range(5):

                text_tok_tar = batch[f'text_tok_tar{option}'].long().to(device)
                text_mask_tar = batch[f'text_mask_tar{option}'].long().to(device)

                #==================================================================================#
                #                            DYNAMIC BATCHING - ENCODER                            #
                #==================================================================================#
                max_tok_len_in_batch = text_mask_tar.sum(1).max().item()
                text_mask_tar = text_mask_tar[:, :max_tok_len_in_batch]
                text_tok_tar = text_tok_tar[:, :max_tok_len_in_batch]

                label = cc( text_tok_tar )
                label[label == 0] = -100

                decoder_out = model(src_text_tok = text_tok_src, src_text_mask = text_mask_src, 
                                    src_event_loc = event_loc_src, src_event_pad_mask = event_mask_src, 
                                    tar_text_tok = text_tok_tar, tar_pad_mask = text_mask_tar)                                    
                answer_per_line = np.append(answer_per_line, decoder_out.loss.item())
    
            final_answers = np.append(final_answers, answer_per_line)

        print(f">>-- {final_answers.reshape(-1, 5).shape[0]}")
        assert final_answers.reshape(-1, 5).shape[0] == len(test_ds)
        acc = sum(final_answers.reshape(-1, 5).argmin(1) == 0)/len(final_answers.reshape(-1, 5).argmin(-1))
        print(f">> The accuracy on {data_path} = {acc}")