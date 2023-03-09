import torch
from transformers import  BertModel, EncoderDecoderModel
from copy import deepcopy as cc
from TreeBERT import *
from torch import nn

class TreeBERT2BERT(nn.Module):

    def __init__(self, config, args):
        super(TreeBERT2BERT, self).__init__()

        self.config = config
        self.encoder = Tree_BERT(self.config)
        self.decoder = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased').decoder
        self.args = args

        if self.args.enc_loading_checkpoint:
            checkpoint = torch.load(f = self.args.enc_loading_checkpoint, map_location=torch.device('cpu'))
            self.encoder.load_state_dict(checkpoint['model'], strict=False)
            print(f'>> Finished loading weights to ENCODER from {self.args.enc_loading_checkpoint}!!\n')
            del checkpoint
        else:    
            BERT = BertModel.from_pretrained('bert-base-uncased')
            self.encoder.load_state_dict(BERT.state_dict(), strict = False);
            del BERT
        
        if self.args.dec_loading_checkpoint:
            checkpoint = torch.load(f = self.args.dec_loading_checkpoint, map_location=torch.device('cpu'))
            self.decoder.load_state_dict(checkpoint['model'], strict=False)
            print(f'>> Finished loading weights to DECODER from {self.args.dec_loading_checkpoint}!!\n')
            del checkpoint
    
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

        encoded_src = encoded_src['hidden_states']
        if self.args.et_sep == True:
            encoded_src = encoded_src[torch.arange(encoded_src.shape[0]).unsqueeze(-1), src_event_loc]
            src_text_mask = cc(src_event_pad_mask)

        # Pass the encoded input to the decoder
        decoder_out = self.decoder(input_ids = tar_text_tok,
                            attention_mask = tar_pad_mask,
                            encoder_hidden_states = encoded_src,
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


class BERT2BERT(nn.Module):

    def __init__(self, args):
        super(BERT2BERT, self).__init__()
        self.args = args
        model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
        self.encoder = cc(model.encoder)
        self.decoder = cc(model.decoder)

        if self.args.enc_loading_checkpoint:
            checkpoint = torch.load(f = self.args.enc_loading_checkpoint, map_location=torch.device('cpu'))
            self.encoder.load_state_dict(checkpoint['model'], strict=False)
            print(f'>> Finished loading weights to ENCODER from {self.args.enc_loading_checkpoint}!!\n')
            del checkpoint        

        if self.args.dec_loading_checkpoint:
            checkpoint = torch.load(f = self.args.dec_loading_checkpoint, map_location=torch.device('cpu'))
            self.decoder.load_state_dict(checkpoint['model'], strict=False)
            print(f'>> Finished loading weights to DECODER from {self.args.dec_loading_checkpoint}!!\n')
            del checkpoint
            
    def forward(self, src_text_tok, src_text_mask, src_event_loc, src_event_pad_mask, tar_text_tok, tar_pad_mask):

        # indices with label -100 will be ignored in the loss computation
        decoder_label = cc(tar_text_tok)
        decoder_label[tar_text_tok == 0] = -100

        encoded_src = self.encoder(input_ids = src_text_tok,
                                         attention_mask = src_text_mask)
        encoded_src = encoded_src['last_hidden_state']        
        if self.args.et_sep == True:
            encoded_src = encoded_src[torch.arange(encoded_src.shape[0]).unsqueeze(-1), src_event_loc]
            src_text_mask = cc(src_event_pad_mask)

        decoder_out = self.decoder(input_ids = tar_text_tok,
                                   attention_mask = tar_pad_mask,
                                   encoder_hidden_states = encoded_src,
                                   encoder_attention_mask = src_text_mask,
                                   labels = decoder_label)
        return decoder_out