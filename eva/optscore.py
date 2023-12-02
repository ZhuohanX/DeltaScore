import torch
import torch.nn as nn
import traceback
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel, GPTJForCausalLM, AutoTokenizer
from typing import List
import numpy as np


class OPTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/opt-350m', type='350m', batch_size=10):
        # Set up model
        self.device = device
        self.type = type
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir='')
        self.model = OPTForCausalLM.from_pretrained(checkpoint, cache_dir='', device_map='auto')
        self.model.eval()
        self.model.to(device)


    def score(self, srcs, tgts, prompt_text, batch_size):
        """ Score a batch of examples """

        def trunk_input(inputs, outputs, reduce_seq, max_length):
            input_ids = self.tokenizer.encode(inputs)[1:-1]
            output_ids = self.tokenizer.encode(outputs)[1:-1]
            reduce_seq_ids = self.tokenizer.encode(reduce_seq)[1:-1]
            total_len = len(input_ids) + len(output_ids)
            if total_len > max_length:
                del_len = len(input_ids) + len(output_ids) - max_length
                reduce_seq_ids = reduce_seq_ids[:len(reduce_seq_ids) - del_len]
                reduce_seq = self.tokenizer.decode(reduce_seq_ids[1:-1])
            return reduce_seq

        score_list = []
        for i,(src, tgt) in enumerate(zip(srcs, tgts)):
            # print('process:'+str(i) + '/'+str(len(srcs)) )
            new_src = trunk_input(src, tgt, src, max_length=self.max_length)
            src = new_src
            text = src + ' ' + tgt
            if i <1:
                print('text: ', text)
                print('tgt: ', tgt)
            input_ids = self.tokenizer.encode(text)
            tgt_ids = self.tokenizer.encode(tgt)[1:]
            output_ids = [-100] * len(input_ids)
            output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
            input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=output_ids,
                        output_hidden_states=True
                    )
                loss, logits, hidden_states = outputs[0], outputs[1], outputs.hidden_states[0]
                loss = loss.item()
                score = -loss
                score_list.append(score)
                # print('score: ',score)
            except RuntimeError:
                # traceback.print_exc()
                print('input_ids: ',input_ids)
                print('output_ids: ', output_ids)
                print(f'source: {src}')
                print(f'target: {tgt}')
                # exit(0)
        return score_list


