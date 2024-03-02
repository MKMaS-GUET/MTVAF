import torch
# from transformers import DataCollator
from torch.utils.data.dataset import Dataset
from typing import Callable, Dict, List, Optional, Tuple, Union

from transformers.data.data_collator import DataCollator

from models.modeling_bert import BertModel
from models.modeling_roberta import RobertaModel
from apex import amp

import torch.nn.functional as F

class Cutoff:

    def __init__(
        self,
        input_ids,
        token_type_ids,
        attention_masks,
        prefix_guids,
        args,
        model,
    ):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_masks = attention_masks
        self.prefix_guids = prefix_guids
        self.args = args
        self.model = model


    def js_div(p, q):
        m = (p + q) / 2
        a = F.kl_div(p.log(), m, reduction='batchmean')
        b = F.kl_div(q.log(), m, reduction='batchmean')
        jsd = ((a + b) / 2)
        return jsd

    def _resolve_loss_item(self, loss, optimizer):
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def _training_step_with_cutoff(
            self, aug_type
    ):

        # Cut embedding_output and attention mask
        input_ids = self.input_ids.to(self.args.device)
        token_type_ids = self.token_type_ids.to(self.args.device)
        embeds = self.model.get_embedding_output(input_ids=input_ids, token_type_ids=token_type_ids)

        masks = self.attention_masks.to(self.args.device)
        input_lens = torch.sum(masks, dim=1).to(self.args.device)

        if aug_type == 'span_cutoff':
            input_embeds, input_masks = self.generate_span_cutoff_embedding(embeds, masks, input_lens)
        elif aug_type == 'token_cutoff':
            input_embeds, input_masks = self.generate_token_cutoff_embedding(embeds, masks, input_lens)
        elif aug_type == 'dim_cutoff':
            input_embeds, input_masks = self.generate_dim_cutoff_embedding(embeds, masks, input_lens)
        else:
            raise NotImplementedError

        cutoff_outputs = self.model.get_bert_output(embedding_output=input_embeds, attention_mask=input_masks, past_key_values=self.prefix_guids)  #重新获取logits
        return cutoff_outputs
        # if self.args.aug_ce_loss > 0:
        #     self.loss += self.args.aug_ce_loss * cutoff_outputs[0]
        '''
        self.loss += cutoff_outputs[0]
        # if self.args.aug_js_loss > 0:
        assert self.args.n_gpu == 1
        ori_logits = self.logits
        aug_logits = cutoff_outputs[1]
        p = torch.softmax(ori_logits + 1e-10, dim=1)
        q = torch.softmax(aug_logits + 1e-10, dim=1)
        aug_js_loss = self.js_div(p, q)  
        # self.loss += self.args.aug_js_loss * aug_js_loss  #+= Jensen-Shannon (JS) Divergence consistency loss
        self.loss += aug_js_loss
        return self.loss
        '''
    def generate_span_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):  # [bsz]个
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            start = int(torch.rand(1).to(self.args.device) * (input_lens[i] - cutoff_length))
            cutoff_embed = torch.cat((embeds[i][:start],
                                      torch.zeros([cutoff_length, embeds.shape[-1]],  # [len,dim]
                                                  dtype=torch.float).to(self.args.device),
                                      embeds[i][start + cutoff_length:]), dim=0)
            cutoff_mask = torch.cat((masks[i][:start],
                                     torch.zeros([cutoff_length], dtype=torch.long).to(self.args.device),
                                     masks[i][start + cutoff_length:]), dim=0)
            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks


    def generate_token_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_length = int(input_lens[i] * self.args.aug_cutoff_ratio)
            zero_index = torch.randint(input_lens[i], (cutoff_length,))

            cutoff_embed = embeds[i]
            cutoff_mask = masks[i]

            tmp_mask = torch.ones(cutoff_embed.shape[0], ).to(self.args.device)  # randn token index mask = 0
            for ind in zero_index:
                tmp_mask[ind] = 0

            cutoff_embed = torch.mul(tmp_mask[:, None], cutoff_embed)  # [L,1] * [L,d]
            cutoff_mask = torch.mul(tmp_mask, cutoff_mask).type(torch.int64)

            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)

        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks


    def generate_dim_cutoff_embedding(self, embeds, masks, input_lens):
        input_embeds = []
        input_masks = []
        for i in range(embeds.shape[0]):
            cutoff_embed = embeds[i]
            cutoff_mask = masks[i]

            cutoff_length = int(cutoff_embed.shape[1] * self.args.aug_cutoff_ratio)  # d * 0.1
            zero_index = torch.randint(cutoff_embed.shape[1], (cutoff_length,))

            tmp_mask = torch.ones(cutoff_embed.shape[1], ).to(self.args.device)
            for ind in zero_index:
                tmp_mask[ind] = 0.

            cutoff_embed = torch.mul(tmp_mask, cutoff_embed)  # [d] * [L,d] 广播

            input_embeds.append(cutoff_embed)
            input_masks.append(cutoff_mask)
        input_embeds = torch.stack(input_embeds, dim=0)
        input_masks = torch.stack(input_masks, dim=0)

        return input_embeds, input_masks