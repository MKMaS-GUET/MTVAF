import copy

import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF

from modules.augument import Cutoff
from modules.parallel import DataParallelCriterion
from .modeling_bert import BertModel
from .modeling_roberta import RobertaModel
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torch.autograd import Variable

class myResnet(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size=7):
        # x shape batch_size * channels * 224 * 224
        # 32 * 3 * 224 * 224

        # batch_size * channels * 112 * 112
        # 32 * 64 * 112 * 112
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # 32 * 256 * 56 * 56
        x = self.resnet.maxpool(x)

        # 32 * 512 * 56 * 56
        x = self.resnet.layer1(x)
        # 32 * 512 * 28 * 28
        x = self.resnet.layer2(x)
        # 32 * 1024 * 14 * 14
        x = self.resnet.layer3(x)
        # 32 * 2048 * 7 * 7
        x = self.resnet.layer4(x)

        # 32 * 2048
        fc = x.mean(3).mean(2)
        # 32 * 2048 * 7 * 7
        att = F.adaptive_avg_pool2d(x, [att_size, att_size])

        # 32 * 2048 * 1 * 1
        x = self.resnet.avgpool(x)
        # 32 * 2048 * 2048
        x = x.view(x.size(0), -1)

        if not self.if_fine_tune:
            x = Variable(x.data)
            fc = Variable(fc.data)
            att = Variable(att.data)

        return x, fc, att

class ImageModel(nn.Module):
    def __init__(self,use_152=False, use_101=False, use_34=False, use_18=False, resnet_root=None):
        super(ImageModel, self).__init__()
        if use_152:
            # self.resnet = resnet152(pretrained=True)
            self.resnet = resnet152(pretrained=False)
            self.resnet.load_state_dict(torch.load(resnet_root+ '/resnet152.pth'))
        elif use_101:
            # self.resnet = resnet101(pretrained=True)
            self.resnet = resnet101(pretrained=False)
            self.resnet.load_state_dict(torch.load(resnet_root + '/resnet101.pth'))
        elif use_34:
            # self.resnet = resnet34(pretrained=True)
            self.resnet = resnet34(pretrained=False)
            self.resnet.load_state_dict(torch.load(resnet_root + '/resnet34.pth'))
        elif use_18:
            # self.resnet = resnet18(pretrained=True)
            self.resnet = resnet18(pretrained=False)
            self.resnet.load_state_dict(torch.load(resnet_root + '/resnet18.pth'))
        else:
            # self.resnet = resnet50(pretrained=True)
            self.resnet = resnet50(pretrained=False)
            self.resnet.load_state_dict(torch.load(resnet_root + '/resnet50.pth'))
        # self.resnet = resnet152(pretrained=True) if use_18 else resnet50(pretrained=True)

    def forward(self, x, aux_imgs=None):
        # full image prompt
        prefix_guids = self.get_resnet_prompt(x)

        if aux_imgs is not None:
            aux_prefix_guids = []
            aux_imgs = aux_imgs.permute([1, 0, 2, 3, 4])
            for i in range(len(aux_imgs)):
                aux_prompt_guid = self.get_resnet_prompt(aux_imgs[i])
                aux_prefix_guids.append(aux_prompt_guid)
            return prefix_guids, aux_prefix_guids
        return prefix_guids, None

    def get_resnet_prompt(self, x):
        prefix_guids = []
        for name, layer in self.resnet.named_children():
            if name == 'fc' or name == 'avgpool':  continue
            x = layer(x)
            if 'layer' in name:
                bsz, channel, ft, _ = x.size()
                kernel = ft // 2
                prompt_kv = nn.AvgPool2d(kernel_size=(kernel, kernel), stride=kernel)(x)
                prefix_guids.append(prompt_kv)
        return prefix_guids

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()


def reconstruct(x, ref):
    if len(x.size()) == 1:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        return x.view([batch_size, turn_num])
    elif len(x.size()) == 2:
        batch_size = ref.size()[0]
        turn_num = ref.size()[1]
        sequence_length = x.size()[1]
        return x.view([batch_size, turn_num, sequence_length])
    else:
        raise Exception()

def flatten_emb_by_sentence(emb, emb_mask):
    batch_size = emb.size()[0]
    seq_length = emb.size()[1]
    flat_emb = flatten(emb)
    flat_emb_mask = emb_mask.view([batch_size * seq_length])
    return flat_emb[flat_emb_mask.nonzero().squeeze(), :]

def get_span_representation(span_starts, span_ends, input, input_mask):
    input_mask = input_mask.to(dtype=span_starts.dtype)  # fp16 compatibility
    input_len = torch.sum(input_mask, dim=-1) # [N]
    word_offset = torch.cumsum(input_len, dim=0) # [N]
    word_offset -= input_len

    span_starts_offset = span_starts + word_offset.unsqueeze(1)
    span_ends_offset = span_ends + word_offset.unsqueeze(1)

    span_starts_offset = span_starts_offset.view([-1])  # [N*M]
    span_ends_offset = span_ends_offset.view([-1])

    span_width = span_ends_offset - span_starts_offset + 1
    JR = torch.max(span_width)
    context_outputs = flatten_emb_by_sentence(input, input_mask)

    text_length = context_outputs.size()[0]

    span_indices = torch.arange(JR).unsqueeze(0).to(span_starts_offset.device) + span_starts_offset.unsqueeze(1)  # [N*M, JR]
    span_indices = torch.min(span_indices, (text_length - 1)*torch.ones_like(span_indices))
    span_text_emb = context_outputs[span_indices, :]
    row_vector = torch.arange(JR).to(span_width.device)
    span_mask = row_vector < span_width.unsqueeze(-1)   # [N*M, JR]
    return span_text_emb, span_mask

def get_self_att_representation(input, input_score, input_mask):
    input_mask = input_mask.to(dtype=input_score.dtype)  # fp16 compatibility
    input_mask = (1.0 - input_mask) * -10000.0
    input_score = input_score + input_mask
    input_prob = nn.Softmax(dim=-1)(input_score)
    input_prob = input_prob.unsqueeze(-1)
    output = torch.sum(input_prob * input, dim=1)
    return output

def distant_cross_entropy(logits, positions, mask=None):
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

class TVNetSAModel(nn.Module):
    def __init__(self, label_list, tokenizer, args, type_num=None, use_weight=False):
        super(TVNetSAModel, self).__init__()
        self.args = args
        self.type_num = type_num
        self.tokenizer = tokenizer
        self.prefix_dim = args.prefix_dim
        self.prefix_len = args.prefix_len
        if "roberta" in args.bert_name:
            self.bert = RobertaModel.from_pretrained(args.bert_name)
        else:
            self.bert = BertModel.from_pretrained(args.bert_name)

        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()
        self.unary_affine = nn.Linear(self.bert.config.hidden_size, 1)
        self.binary_affine = nn.Linear(self.bert.config.hidden_size, 2)

        self.num_labels = len(label_list) + 1
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4)

        if args.use_prefix:
            if self.args.use_152:
                self.image_model = ImageModel(use_152=True, resnet_root=self.args.resnet_root)
            elif self.args.use_101:
                self.image_model = ImageModel(use_101=True, resnet_root=self.args.resnet_root)
            elif self.args.use_34:
                self.image_model = ImageModel(use_34=True, resnet_root=self.args.resnet_root)
            elif self.args.use_18:
                self.image_model = ImageModel(use_18=True, resnet_root=self.args.resnet_root)
            else:
                self.image_model = ImageModel(resnet_root=self.args.resnet_root)
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=3840, out_features=800),
                nn.Tanh(),
                nn.Linear(in_features=800, out_features=4 * 2 * 768)
            )
            self.projectors = nn.ModuleList([nn.Linear(4 * 768 * 2, 4) for i in range(12)])

        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        if args.gcn_layer_number > 0:
            self.gcn = DiGCNModuleAtt(args.gcn_layer_number, self.bert.config.hidden_size, use_weight=use_weight, output_all_layers=False)
            self.dep_embedding = nn.Embedding(type_num, self.bert.config.hidden_size, padding_idx=0)
        if args.num_layers > 0:
            self.gcn = GCNBert(self.bert, self.args, self.args.num_layers)
        if self.args.use_probe:
            from probes.probe_trainModel import probe
            from probes.loss import CombineLoss
            self.oneWordpsdProbe = probe(
                args={'probe': {'maximum_rank': int(768 / 2)}, 'model': {'hidden_dim': 768}})
            for key, value in self.oneWordpsdProbe.named_parameters():
                value.requires_grad = True
            self.combineLoss = CombineLoss(args.beta)
    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None,
            start_positions=None, end_positions=None, span_starts=None, span_ends=None,
            polarity_labels=None, label_masks=None, images=None, aux_imgs=None,
            valid_ids=None, adjacency_matrix=None, output_attention=False, augument=False, labels=None,
            adj_matrix=None, src_mask=None, aspect_mask=None, polaritys = None
    ):

        bsz = input_ids.size(0)
        if self.args.use_prefix:
            prefix_guids = self.get_visual_prompt(images, aux_imgs)
            prefix_guids_length = prefix_guids[0][0].shape[2]
            prefix_guids_mask = torch.ones((bsz, prefix_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prefix_guids_mask, attention_mask), dim=1)
        else:
            prefix_guids = None
            prompt_attention_mask = attention_mask
        if self.args.use_probe:
            start_logits, end_logits, sequence_output, prob_loss = self.extraction(prompt_attention_mask, input_ids, prefix_guids, token_type_ids, augument, labels)
        elif self.args.num_layers > 0:
            gcn_logits, penal, start_logits, end_logits, sequence_output = self.extraction(prompt_attention_mask, input_ids,prefix_guids, token_type_ids,augument, labels, adj_matrix, src_mask, aspect_mask)
        else:
            start_logits, end_logits, sequence_output = self.extraction(prompt_attention_mask, input_ids,prefix_guids, token_type_ids,augument, labels)

        if self.args.gcn_layer_number > 0:
            if valid_ids is not None:
                batch_size, max_len, feat_dim = sequence_output.shape
                valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=self.args.device)
                for i in range(batch_size):
                    temp = sequence_output[i][valid_ids[i] == 1]
                    valid_output[i][:temp.size(0)] = temp
            else:
                valid_output = sequence_output
            sequence_output = self.dropout(valid_output)

            sequence_output = self.gcn(sequence_output, adjacency_matrix, output_attention=output_attention)
            if output_attention is True:
                sequence_output, output_attention_list = sequence_output

        logits, ac_logits = self.classification(attention_mask=attention_mask, span_starts=span_starts,span_ends=span_ends,\
                                       sequence_input=sequence_output)

        ac_loss_fct = nn.CrossEntropyLoss()
        flat_polarity_labels = flatten(polarity_labels)
        flat_label_masks = flatten(label_masks).to(dtype=ac_logits.dtype)
        if self.args.n_gpu > 1:
            criterion = CriterionLoss()
            criterion = DataParallelCriterion(criterion)
            criterion.to(self.args.device)
            tot_loss = criterion(start_logits, start_positions, end_logits, end_positions, ac_logits, flat_polarity_labels, flat_label_masks)

        else:
            start_loss = distant_cross_entropy(start_logits, start_positions)
            end_loss = distant_cross_entropy(end_logits, end_positions)
            ae_loss = (start_loss + end_loss) / 2

            ac_loss = ac_loss_fct(ac_logits, flat_polarity_labels)
            ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

            tot_loss = ae_loss + ac_loss

        if self.args.num_layers > 0:
            tat_loss = ac_loss_fct(gcn_logits, polaritys)
            tot_loss = tot_loss + tat_loss + penal

        if self.args.use_probe:
            combine_loss = self.combineLoss(tot_loss, prob_loss, self.args.num_epochs)
            return TokenClassifierOutput(
                loss=combine_loss,
                logits=logits
            ), prob_loss, tot_loss
        else:
            return TokenClassifierOutput(
                loss=tot_loss,
                logits=logits
            )

    def extraction(self, prompt_attention_mask, input_ids, prefix_guids, token_type_ids, augument=False, labels=None, adj_matrix=None, src_mask=None, aspect_mask=None):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                token_type_ids=token_type_ids,
                                past_key_values=prefix_guids,
                                output_attentions=True,
                                output_hidden_states=True,
                                return_dict=True)
        output_hidden_states = bert_output['hidden_states'][7]

        if augument:
            cutoff = Cutoff(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_masks=prompt_attention_mask,
                prefix_guids=prefix_guids,
                args=self.args,
                model=self.bert,
            )
            cutoff_outputs = cutoff._training_step_with_cutoff(self.args.aug_type)
            sequence_output = self.dropout(cutoff_outputs[0])
        else:
            sequence_output = bert_output['last_hidden_state']
            sequence_output = self.dropout(sequence_output)

        if self.args.num_layers > 0:
            gcn_logits, penal = self.gcn(adj_matrix, src_mask, aspect_mask, bert_output['last_hidden_state'], bert_output['pooler_output'])

        ae_logits = self.binary_affine(sequence_output)
        start_logits, end_logits = ae_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if self.args.use_probe:
            prob_loss = self.oneWordpsdProbe(output_hidden_states).to("cuda:0" if torch.cuda.is_available() else "cpu")
            return start_logits, end_logits, sequence_output, prob_loss
        elif self.args.num_layers > 0:
            return gcn_logits, penal, start_logits, end_logits, sequence_output
        else:
            return start_logits, end_logits, sequence_output

    def classification(self, span_starts, span_ends, sequence_input, attention_mask):
        span_output, span_mask = get_span_representation(span_starts, span_ends, sequence_input,
                                                         attention_mask)

        span_score = self.unary_affine(span_output)
        span_score = span_score.squeeze(-1)  # [N*M, JR]
        span_pooled_output = get_self_att_representation(span_output, span_score, span_mask)  # [N*M, D]

        span_pooled_output = self.dense(span_pooled_output)
        span_pooled_output = self.activation(span_pooled_output)
        span_pooled_output = self.dropout(span_pooled_output)
        ac_logits = self.classifier(span_pooled_output)  # [N*M, 5]

        return reconstruct(ac_logits, span_starts), ac_logits


    def get_visual_prompt(self, images, aux_imgs):
        bsz = images.size(0)
        prefix_guids, aux_prefix_guids = self.image_model(images, aux_imgs)

        prefix_guids = torch.cat(prefix_guids, dim=1).view(bsz, self.args.prefix_len, -1)
        aux_prefix_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prefix_len, -1) for aux_prompt_guid in aux_prefix_guids]  # 3 x [bsz, 4, 3840]

        prefix_guids = self.encoder_conv(prefix_guids)
        aux_prefix_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prefix_guids]
        split_prefix_guids = prefix_guids.split(768*2, dim=-1)
        split_aux_prefix_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prefix_guids]

        result = []
        for idx in range(12):
            sum_prefix_guids = torch.stack(split_prefix_guids).sum(0).view(bsz, -1) / 4
            prefix_projector = F.softmax(F.leaky_relu(self.projectors[idx](sum_prefix_guids)), dim=-1)

            key_val = torch.zeros_like(split_prefix_guids[0]).to(self.args.device)
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prefix_projector[:, i].view(-1, 1), split_prefix_guids[i])

            aux_key_vals = []
            for split_aux_prompt_guid in split_aux_prefix_guids:
                sum_aux_prefix_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4
                aux_prefix_projector = F.softmax(F.leaky_relu(self.projectors[idx](sum_aux_prefix_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prefix_projector[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()
            temp_dict = (key, value)
            result.append(temp_dict)
        return result

class TVNetSAModel2(nn.Module):
    def __init__(self, label_list, tokenizer, args, type_num=None, use_weight=False):
        super(TVNetSAModel2, self).__init__()
        self.args = args
        self.type_num = type_num
        self.tokenizer = tokenizer
        self.prefix_dim = args.prefix_dim
        self.prefix_len = args.prefix_len

        if "roberta" in args.bert_name:
            self.bert = RobertaModel.from_pretrained(args.bert_name)
            # self.self_image_attention = RobertaSelfEncoder(self.bert.config)
        else:
            self.bert = BertModel.from_pretrained(args.bert_name)
            # self.self_image_attention = BertSelfEncoder(self.bert.config)

        self.num_labels = len(label_list) + 1
        # self.classifier = nn.Linear(self.bert.config.hidden_size, 5)

        if args.use_prefix:
            if self.args.use_152:
                self.image_model = ImageModel(use_152=True, resnet_root=self.args.resnet_root)
            elif self.args.use_101:
                self.image_model = ImageModel(use_101=True, resnet_root=self.args.resnet_root)
            elif self.args.use_34:
                self.image_model = ImageModel(use_34=True, resnet_root=self.args.resnet_root)
            elif self.args.use_18:
                self.image_model = ImageModel(use_18=True, resnet_root=self.args.resnet_root)
            else:
                self.image_model = ImageModel(resnet_root=self.args.resnet_root)
            self.encoder_conv = nn.Sequential(
                nn.Linear(in_features=960, out_features=800),
                nn.Tanh(),
                nn.Linear(in_features=800, out_features=4 * 2 * 768)
            ) if self.args.use_34 or self.args.use_18 else nn.Sequential(
                nn.Linear(in_features=3840, out_features=800),
                nn.Tanh(),
                nn.Linear(in_features=800, out_features=4 * 2 * 768)
            )
            self.projectors = nn.ModuleList([nn.Linear(4 * 768 * 2, 4) for i in range(12)])

            self.img_dropout = nn.Dropout(0.2)
            self.img_classifier = nn.Linear(4 * 2 * 768, 2089)  # ANP分类器
            self.aux_img_classifier = nn.ModuleList([nn.Linear(4 * 2 * 768, 2089) for i in range(3)])
            self.klloss = nn.KLDivLoss(reduction='batchmean')
            self.aux_klloss = [nn.KLDivLoss(reduction='batchmean') for j in range(3)]

        print("label_list/num_labels",self.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        if self.args.use_probe:
            import sys
            sys.path.append(os.path.abspath('..'+'/HVPNeT-main/models'))
            from probes.probe_trainModel import probe
            from probes.loss import CombineLoss
            self.oneWordpsdProbe = probe(
                args={'probe': {'maximum_rank': int(768 / 2)}, 'model': {'hidden_dim': 768}})
            self.oneWordpsdProbe.load_state_dict(torch.load(
                os.path.abspath('.') + '/models/psdProbe_base_savel{:}.pt'.format(7), map_location="cpu").state_dict())
            for key, value in self.oneWordpsdProbe.named_parameters():
                value.requires_grad = True
            self.combineLoss = CombineLoss(args.beta)

    def forward(
            self, input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None, imagelabel=None, images=None, aux_imgs=None,
    ):

        bsz = input_ids.size(0)
        img_tag_loss = 0
        if self.args.use_prefix:
            prefix_guids, img_tag_loss, aux_img_tag_loss = self.get_visual_prompt(images, aux_imgs, imagelabel)
            img_tag_loss = img_tag_loss if self.args.noauxloss else img_tag_loss + sum(aux_img_tag_loss)
            prefix_guids_length = prefix_guids[0][0].shape[2]
            prefix_guids_mask = torch.ones((bsz, prefix_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prefix_guids_mask, attention_mask), dim=1)
        else:
            prefix_guids = None
            prompt_attention_mask = attention_mask
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=prompt_attention_mask,
                                token_type_ids=token_type_ids,
                                past_key_values=prefix_guids,
                                output_attentions=True,
                                output_hidden_states=True,
                                return_dict=True)
        output_hidden_states = bert_output['hidden_states'][7]

        sequence_output = bert_output['last_hidden_state']
        sequence_output = self.dropout(sequence_output)
        if self.args.use_probe:
            prob_loss = self.oneWordpsdProbe(output_hidden_states).to("cuda:0" if torch.cuda.is_available() else "cpu")

        emissions = self.fc(sequence_output)
        logits = self.crf.decode(emissions, attention_mask.byte())

        loss=None
        if labels is not None:
            if self.args.n_gpu > 1:
                criterion = CriterionLoss2(self.crf)  # CriterionCrossEntropy()
                criterion = DataParallelCriterion(criterion)
                criterion.to(self.args.device)
                loss = criterion(emissions, labels, attention_mask)
            else:
                loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
        if self.args.use_probe:
            combine_loss = self.combineLoss(loss, prob_loss, 30)
            return TokenClassifierOutput(
                loss=combine_loss + self.args.alpha * img_tag_loss,
                logits=logits
            ), prob_loss, self.args.alpha * img_tag_loss
        else:
            return TokenClassifierOutput(
                loss=loss + self.args.alpha * img_tag_loss,
                logits=logits
            )

    def get_visual_prompt(self, images, aux_imgs, imagelabel):
        bsz = images.size(0)
        prefix_guids, aux_prefix_guids = self.image_model(images, aux_imgs)

        prefix_guids = torch.cat(prefix_guids, dim=1).view(bsz, self.args.prefix_len, -1)
        aux_prefix_guids = [torch.cat(aux_prompt_guid, dim=1).view(bsz, self.args.prefix_len, -1) for aux_prompt_guid in aux_prefix_guids]  # 3 x [bsz, 4, 3840]

        prefix_guids = self.encoder_conv(prefix_guids)
        aux_prefix_guids = [self.encoder_conv(aux_prompt_guid) for aux_prompt_guid in aux_prefix_guids]

        split_prefix_guids = prefix_guids.split(768*2, dim=-1)
        split_aux_prefix_guids = [aux_prompt_guid.split(768*2, dim=-1) for aux_prompt_guid in aux_prefix_guids]

        img_tag_loss = 0
        aux_img_tag_loss = []
        if (self.args.vao):
            prefix_guids_mean = prefix_guids.mean(dim= 1)
            prefix_guids_dropout = self.img_dropout(prefix_guids_mean)
            img_tag = self.img_classifier(prefix_guids_dropout)
            img_tag_softmax = F.softmax(img_tag, dim=-1)
            img_tag_loss = self.klloss(img_tag_softmax.log(), imagelabel.to(self.args.device))
            k = 0
            for aux_prompt_guid in aux_prefix_guids:
                aux_prompt_guid_mean = aux_prompt_guid.mean(dim=1)
                aux_prompt_guid_dropout = self.img_dropout(aux_prompt_guid_mean)
                aux_img_tag = self.aux_img_classifier[k](aux_prompt_guid_dropout)
                aux_img_tag_softmax = F.softmax(aux_img_tag, dim=-1)
                temp_img_tag_loss = self.aux_klloss[k](aux_img_tag_softmax.log(), imagelabel.to(self.args.device))
                aux_img_tag_loss.append(temp_img_tag_loss)
                k += 1

        result = []
        for idx in range(12):
            sum_prefix_guids = torch.stack(split_prefix_guids).sum(0).view(bsz, -1) / 4
            prefix_projector = F.softmax(F.leaky_relu(self.projectors[idx](sum_prefix_guids)), dim=-1)

            key_val = torch.zeros_like(split_prefix_guids[0]).to(self.args.device)
            for i in range(4):
                key_val = key_val + torch.einsum('bg,blh->blh', prefix_projector[:, i].view(-1, 1), split_prefix_guids[i])

            aux_key_vals = []
            for split_aux_prompt_guid in split_aux_prefix_guids:
                sum_aux_prefix_guids = torch.stack(split_aux_prompt_guid).sum(0).view(bsz, -1) / 4
                aux_prefix_projector = F.softmax(F.leaky_relu(self.projectors[idx](sum_aux_prefix_guids)), dim=-1)
                aux_key_val = torch.zeros_like(split_aux_prompt_guid[0]).to(self.args.device)
                for i in range(4):
                    aux_key_val = aux_key_val + torch.einsum('bg,blh->blh', aux_prefix_projector[:, i].view(-1, 1), split_aux_prompt_guid[i])
                aux_key_vals.append(aux_key_val)
            key_val = [key_val] + aux_key_vals
            key_val = torch.cat(key_val, dim=1)
            key_val = key_val.split(768, dim=-1)
            key, value = key_val[0].reshape(bsz, 12, -1, 64).contiguous(), key_val[1].reshape(bsz, 12, -1, 64).contiguous()
            temp_dict = (key, value)
            result.append(temp_dict)
        return result, img_tag_loss, aux_img_tag_loss

class Distant_CE(nn.Module):
    def __init__(self):
        super(Distant_CE, self).__init__()
        self.criterion = nn.LogSoftmax(dim=-1)

    def distant_cross_entropy(self, logits, positions, mask=None):
        log_probs = self.criterion(logits)
        if mask is not None:
            loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                                   (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(
                                       dtype=log_probs.dtype)))
        else:
            loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                                   torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
        return loss

    def forward(self, logits, positions):
        return distant_cross_entropy(logits, positions)

class CriterionLoss(nn.Module):

    def __init__(self):
        super(CriterionLoss, self).__init__()
        self.criterion1 = Distant_CE()
        self.criterion2 = nn.CrossEntropyLoss()

    def forward(self, start_logits, start_positions, end_logits, end_positions, ac_logits, flat_polarity_labels, flat_label_masks):

        start_loss = self.criterion1(start_logits, start_positions)
        end_loss = self.criterion1(end_logits, end_positions)
        ae_loss = (start_loss + end_loss) / 2

        ac_loss = self.criterion2(ac_logits, flat_polarity_labels)
        ac_loss = torch.sum(flat_label_masks * ac_loss) / flat_label_masks.sum()

        return ae_loss + ac_loss

class CriterionLoss2(nn.Module):

    def __init__(self, crf):
        super(CriterionLoss, self).__init__()
        self.crf = crf

    def forward(self, emissions, labels, attention_mask):
        return -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
############################################################ D-GCN #########################
import torch.nn.init as init
import math
try:
    import apex
    #apex.amp.register_half_function(apex.normalization.fused_layer_norm, 'FusedLayerNorm')
    import apex.normalization
    #apex.amp.register_float_function(apex.normalization.FusedLayerNorm, 'forward')
    BertLayerNorm = apex.normalization.FusedLayerNorm
except ImportError:
    # print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class DiGCNLayerAtt(nn.Module):
    def __init__(self, hidden_size, use_weight=False):
        super(DiGCNLayerAtt, self).__init__()
        self.temper = hidden_size ** 0.5
        self.use_weight = use_weight
        self.relu = nn.ReLU()
        self.relu_left = nn.ReLU()
        self.relu_self = nn.ReLU()
        self.relu_right = nn.ReLU()

        self.linear = nn.Linear(hidden_size, hidden_size)

        self.left_linear = nn.Linear(hidden_size, hidden_size)
        self.right_linear = nn.Linear(hidden_size, hidden_size)
        self.self_linear = nn.Linear(hidden_size, hidden_size)

        self.output_layer_norm = BertLayerNorm(hidden_size)

        self.reset_parameters(self.linear)
        self.reset_parameters(self.left_linear)
        self.reset_parameters(self.right_linear)
        self.reset_parameters(self.self_linear)

        self.softmax = nn.Softmax(dim=-1)

    def reset_parameters(self, linear):
        init.xavier_normal_(linear.weight)
        # init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(linear.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(linear.bias, -bound, bound)

    def get_att(self, matrix_1, matrix_2, adjacency_matrix):
        u = torch.matmul(matrix_1.float(), matrix_2.permute(0, 2, 1).float()) / self.temper
        attention_scores = self.softmax(u)
        delta_exp_u = torch.mul(attention_scores, adjacency_matrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)
        attention = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10).type_as(matrix_1)
        return attention

    def forward(self, hidden_state, adjacency_matrix, output_attention=False):
        context_attention = self.get_att(hidden_state, hidden_state, adjacency_matrix)

        hidden_state_left = self.left_linear(hidden_state)
        hidden_state_self = self.self_linear(hidden_state)
        hidden_state_right = self.right_linear(hidden_state)

        context_attention_left = torch.triu(context_attention, diagonal=1)
        context_attention_self = torch.triu(context_attention, diagonal=0) - context_attention_left
        context_attention_right = context_attention - torch.triu(context_attention, diagonal=0)

        context_attention = torch.bmm(context_attention_left.float(), hidden_state_left.float()) \
                            + torch.bmm(context_attention_self.float(), hidden_state_self.float()) \
                            + torch.bmm(context_attention_right.float(), hidden_state_right.float())

        output_attention_list = [context_attention_left, context_attention_self, context_attention_right]


        o = self.output_layer_norm(context_attention.type_as(hidden_state))
        output = self.relu(o).type_as(hidden_state)

        if output_attention is True:
            return (output, output_attention_list)
        return output

class DiGCNModuleAtt(nn.Module):
    def __init__(self, layer_number, hidden_size, use_weight=False, output_all_layers=False):
        super(DiGCNModuleAtt, self).__init__()
        if layer_number < 1:
            raise ValueError()
        self.layer_number = layer_number
        self.output_all_layers = output_all_layers
        self.GCNLayers = nn.ModuleList(([DiGCNLayerAtt(hidden_size, use_weight)
                                         for _ in range(self.layer_number)]))

    def forward(self, hidden_state, adjacency_matrix, output_attention=False):
        # hidden_state = self.first_GCNLayer(hidden_state, adjacency_matrix, type_seq, type_matrix)
        # all_output_layers.append(hidden_state)

        all_output_layers = []

        output_attention_list = []
        for gcn in self.GCNLayers:
            hidden_state = gcn(hidden_state, adjacency_matrix, output_attention=output_attention)
            if output_attention is True:
                hidden_state, output_attention_list = hidden_state
            all_output_layers.append(hidden_state)

        if self.output_all_layers:
            if output_attention is True:
                return all_output_layers, output_attention_list
            return all_output_layers
        else:
            if output_attention is True:
                return all_output_layers[-1], output_attention_list
            return all_output_layers[-1]

class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = bert.config.hidden_size // 2
        self.attention_heads = bert.config.num_attention_heads
        self.bert_dim = bert.config.hidden_size
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(bert.config.hidden_size)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(self.attention_heads, self.bert_dim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.dualgcn_classifier = nn.Linear(bert.config.hidden_size * 2, 3)

    def forward(self, adj, src_mask, aspect_mask,sequence_output, pooled_output):
        src_mask = src_mask.unsqueeze(-2)

        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]

        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag /= self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).to(self.opt.device)
        adj_ag = src_mask.transpose(1, 2) * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        gcn_inputs = self.bert_drop(sequence_output)
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs

        for l in range(self.layers):
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.relu(AxW_ag)

            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag

        # avg pooling asp feature
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.mem_dim)
        outputs1 = (outputs_ag * aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn

        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.dualgcn_classifier(final_outputs)
        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).to(self.opt.device)
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).to(self.opt.device)

        # penal = None
        penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).to(self.opt.device)
        penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj)).to(self.opt.device)
        penal = self.opt.alpha * penal1 + self.opt.beta * penal2

        return logits, penal

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn