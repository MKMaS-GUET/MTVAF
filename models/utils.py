import json
import collections
import logging
import random

import numpy as np

from squad.squad_utils import  get_final_text, _get_best_indexes
from squad.squad_evaluate import exact_match_score, f1_score
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer,RobertaTokenizer
import os
import torch

label_to_id = {'other': 0, 'neutral': 1, 'positive': 2, 'negative': 3, 'conflict': 4}
id_to_label = {0: 'other', 1: 'neutral', 2: 'positive', 3: 'negative', 4: 'conflict'}
SEP_TAG = "<SEP>"


class SemEvalExample(object):
    def __init__(self,
                    example_id,
                    sent_tokens,
                    term_texts=None,
                    start_positions=None,
                    end_positions=None,
                    polarities=None,
                    image_labels=None,
                    image_ids=None,
                    raw_image_data=None,
                    ):
        self.example_id = example_id
        self.sent_tokens = sent_tokens
        self.term_texts = term_texts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.polarities = polarities
        self.image_labels = image_labels
        self.image_ids = image_ids
        self.raw_image_data = raw_image_data

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        # s += "example_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", sent_tokens: [%s]" % (" ".join(self.sent_tokens))
        if self.term_texts:
            s += ", term_texts: {}".format(self.term_texts)
        # if self.start_positions:
        #     s += ", start_positions: {}".format(self.start_positions)
        # if self.end_positions:
        #     s += ", end_positions: {}".format(self.end_positions)
        if self.polarities:
            s += ", polarities: {}".format(self.polarities)
        return s

class SemEvalExample1(object):
    def __init__(self,
                    example_id,
                    sent_tokens,
                    term_texts=None,
                    start_positions=None,
                    end_positions=None,
                    polarities=None,
                    image_labels=None,
                    image_ids=None,
                    raw_image_data=None,

                    dep=None,
                    adj=None,
                    dep_text=None,
                    adj_matrix=None,
                    ):
        self.example_id = example_id
        self.sent_tokens = sent_tokens
        self.term_texts = term_texts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.polarities = polarities
        self.image_labels = image_labels
        self.image_ids = image_ids
        self.raw_image_data = raw_image_data

        self.dep = dep
        self.adj = adj
        self.dep_text = dep_text
        self.adj_matrix = adj_matrix

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        # s += "example_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", sent_tokens: [%s]" % (" ".join(self.sent_tokens))
        if self.term_texts:
            s += ", term_texts: {}".format(self.term_texts)
        # if self.start_positions:
        #     s += ", start_positions: {}".format(self.start_positions)
        # if self.end_positions:
        #     s += ", end_positions: {}".format(self.end_positions)
        if self.polarities:
            s += ", polarities: {}".format(self.polarities)
        return s

class SemEvalExample2(object):
    def __init__(self,
                    example_id,
                    sent_tokens,
                    term_texts=None,
                    start_positions=None,
                    end_positions=None,
                    polarities=None,
                    image_labels=None,
                    image_ids=None,
                    raw_image_data=None,

                    adj_matrix=None,
                    src_mask=None,
                    aspect_mask=None,
                    polaritys=None,
                    ):
        self.example_id = example_id
        self.sent_tokens = sent_tokens
        self.term_texts = term_texts
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.polarities = polarities
        self.image_labels = image_labels
        self.image_ids = image_ids
        self.raw_image_data = raw_image_data

        self.adj_matrix = adj_matrix
        self.src_mask = src_mask
        self.aspect_mask = aspect_mask
        self.polaritys = polaritys

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        # s += "example_id: %s" % (tokenization.printable_text(self.example_id))
        s += ", sent_tokens: [%s]" % (" ".join(self.sent_tokens))
        if self.term_texts:
            s += ", term_texts: {}".format(self.term_texts)
        # if self.start_positions:
        #     s += ", start_positions: {}".format(self.start_positions)
        # if self.end_positions:
        #     s += ", end_positions: {}".format(self.end_positions)
        if self.polarities:
            s += ", polarities: {}".format(self.polarities)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_positions=None,
                 end_positions=None,
                 start_indexes=None,
                 end_indexes=None,
                 bio_labels=None,
                 polarity_positions=None,
                 polarity_labels=None,
                 label_masks=None,
                 image_labels=None,
                 image_ids=None,
                 raw_image_data=None,

                 src_mask=None,
                 aspect_mask=None,
                 polaritys=None,
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_indexes = start_indexes
        self.end_indexes = end_indexes
        self.bio_labels = bio_labels
        self.polarity_positions = polarity_positions
        self.polarity_labels = polarity_labels
        self.label_masks = label_masks
        self.image_labels = image_labels
        self.image_ids = image_ids
        self.raw_image_data = raw_image_data

        self.src_mask = src_mask
        self.aspect_mask = aspect_mask
        self.polaritys = polaritys

from collections import defaultdict
def get_idx_for_item(lis: list, item: str) -> int:
    ids = [i for i in range(len(lis))]
    idx2item = dict(zip(ids,lis)) #orign
    item2idx = defaultdict(list)
    for k,v in idx2item.items():
        item2idx[v].append(k)  #transpose
#     item = item.encode("utf-8")
    if item in item2idx.keys():
        return item2idx[item]
    else:
        return 0

def replace_sep_token(words: list, args) -> list:
    ids = get_idx_for_item(words, SEP_TAG)
    if "roberta" in args.bert_name:
        for i in range(len(ids)):
            words[ids[i]] = "</s>"
    else:
        for i in range(len(ids)):
            words[ids[i]] = "[SEP]"
    return words

def convert_examples_to_features(examples, tokenizer, max_seq_length, verbose_logging=False, logger=None):
    max_term_num = max([len(example.term_texts) for (example_index, example) in enumerate(examples)])
    max_sent_length, max_term_length = 0, 0

    unique_id = 1000000000
    features = []
    maxs = 0
    src_mask, aspect_mask, polaritys = None, None, None

    for (example_index, example) in enumerate(examples):
        tok_to_orig_index = [] # token to the original index
        orig_to_tok_index = [] # transverse ,orginal first terms
        all_doc_tokens = []
        for (i, token) in enumerate(example.sent_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        if len(all_doc_tokens) > max_sent_length:
            max_sent_length = len(all_doc_tokens)
        # problem here ,finding that the tokenizer transform the terms into segments
        tok_start_positions = []
        tok_end_positions = []
        for start_position, end_position in \
                zip(example.start_positions, example.end_positions):
            tok_start_position = orig_to_tok_index[start_position]
            if end_position < len(example.sent_tokens) - 1:
                tok_end_position = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            tok_start_positions.append(tok_start_position)
            tok_end_positions.append(tok_end_position)

        # Account for [CLS] and [SEP] with "- 2"
        if len(all_doc_tokens) > max_seq_length - 2:
            all_doc_tokens = all_doc_tokens[0:(max_seq_length - 2)]
        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        if isinstance(tokenizer, RobertaTokenizer):
            tokens.append("<s>")
        else:
            tokens.append("[CLS]")
        segment_ids.append(0)

        for index, token in enumerate(all_doc_tokens):
            token_to_orig_map[len(tokens)] = tok_to_orig_index[index]
            tokens.append(token)
            segment_ids.append(0)
        if isinstance(tokenizer, RobertaTokenizer):
            tokens.append("</s>")
        else:
            tokens.append("[SEP]")
        segment_ids.append(0)
        # transpose the tokens into the ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        maxs = len(input_ids) if maxs < len(input_ids) else maxs         ##################################
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # For distant supervision, we annotate the positions of all answer spans
        start_positions = [0] * len(input_ids)
        end_positions = [0] * len(input_ids)
        bio_labels = [0] * len(input_ids)
        polarity_positions = [0] * len(input_ids)
        start_indexes, end_indexes = [], []
        for tok_start_position, tok_end_position, polarity in zip(tok_start_positions, tok_end_positions, example.polarities):
            if (tok_start_position >= 0 and tok_end_position <= (max_seq_length - 1)):
                start_position = tok_start_position + 1   # [CLS]
                end_position = tok_end_position + 1   # [CLS]
                start_positions[start_position] = 1
                end_positions[end_position] = 1
                start_indexes.append(start_position)
                end_indexes.append(end_position)
                term_length = tok_end_position - tok_start_position + 1
                max_term_length = term_length if term_length > max_term_length else max_term_length
                bio_labels[start_position] = 2  # 'B'
                if start_position < end_position:
                    for idx in range(start_position + 1, end_position + 1):
                        bio_labels[idx] = 1  # 'I'
                for idx in range(start_position, end_position + 1):
                    polarity_positions[idx] = label_to_id[polarity]
        polarity_labels = [label_to_id[polarity] for polarity in example.polarities]
        label_masks = [1] * len(polarity_labels)

        while len(start_indexes) < max_term_num:
            start_indexes.append(0)
            end_indexes.append(0)
            polarity_labels.append(0)
            label_masks.append(0)
        image_labels = example.image_labels
        image_ids = example.image_ids
        raw_image_data = example.raw_image_data
        assert len(start_indexes) == max_term_num
        assert len(end_indexes) == max_term_num
        assert len(polarity_labels) == max_term_num
        assert len(label_masks) == max_term_num

        if example_index < 1 and verbose_logging:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("tokens: {}".format(tokens))
            logger.info("token_to_orig_map: {}".format(token_to_orig_map))
            logger.info("start_indexes: {}".format(start_indexes))
            logger.info("end_indexes: {}".format(end_indexes))
            logger.info("bio_labels: {}".format(bio_labels))
            logger.info("polarity_positions: {}".format(polarity_positions))
            logger.info("polarity_labels: {}".format(polarity_labels))
        # 1:nerutral 3 negative 2:positive
        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_positions=start_positions,
                end_positions=end_positions,
                start_indexes=start_indexes,
                end_indexes=end_indexes,
                bio_labels=bio_labels,
                polarity_positions=polarity_positions,
                polarity_labels=polarity_labels,
                label_masks=label_masks,
                image_labels=image_labels,
                image_ids=image_ids,
                raw_image_data=raw_image_data,
                src_mask=src_mask,
                aspect_mask=aspect_mask,
                polaritys=polaritys,
            ))
        unique_id += 1
    print("maxs:",maxs)
    logger.info("Max sentence length: {}".format(max_sent_length))
    logger.info("Max term length: {}".format(max_term_length))
    logger.info("Max term num: {}".format(max_term_num))
    return features



RawSpanResult = collections.namedtuple("RawSpanResult",
                                       ["unique_id", "start_logits", "end_logits"])

RawSpanCollapsedResult = collections.namedtuple("RawSpanCollapsedResult",
                                       ["unique_id", "neu_start_logits", "neu_end_logits", "pos_start_logits", "pos_end_logits",
                                        "neg_start_logits", "neg_end_logits"])

RawBIOResult = collections.namedtuple("RawBIOResult", ["unique_id", "bio_pred"])

RawBIOClsResult = collections.namedtuple("RawBIOClsResult", ["unique_id", "start_indexes", "end_indexes", "bio_pred", "span_masks"])

RawFinalResult = collections.namedtuple("RawFinalResult",
                                        ["unique_id", "start_indexes", "end_indexes", "cls_pred", "span_masks"])


def wrapped_get_final_text(example, feature, start_index, end_index, do_lower_case, verbose_logging, logger):
    tok_tokens = feature.tokens[start_index:(end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[start_index]
    orig_doc_end = feature.token_to_orig_map[end_index]
    orig_tokens = example.sent_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)

    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")

    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)

    final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging, logger)
    return final_text


def span_annotate_candidates(all_examples, batch_features, batch_results, filter_type, mode, use_heuristics, use_nms,
                             logit_threshold, n_best_size, max_answer_length, do_lower_case, verbose_logging, logger):
    """Annotate top-k candidate answers into features."""
    unique_id_to_result = {}
    for result in batch_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    batch_span_starts, batch_span_ends, batch_labels, batch_label_masks = [], [], [], []
    for (feature_index, feature) in enumerate(batch_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        seen_predictions = {}
        span_starts, span_ends, labels, label_masks = [], [], [], []
        if mode=="train":
            # add ground-truth terms
            for start_index, end_index, polarity_label, mask in \
                    zip(feature.start_indexes, feature.end_indexes, feature.polarity_labels, feature.label_masks):
                if mask and start_index in feature.token_to_orig_map and end_index in feature.token_to_orig_map:
                    final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                        do_lower_case, verbose_logging, logger)
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text] = True

                    span_starts.append(start_index)
                    span_ends.append(end_index)
                    labels.append(polarity_label)
                    label_masks.append(1)
        else:
            prelim_predictions_per_feature = []
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    start_logit = result.start_logits[start_index]
                    end_logit = result.end_logits[end_index]
                    if start_logit + end_logit < logit_threshold:
                        continue

                    prelim_predictions_per_feature.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logit,
                            end_logit=end_logit))

            if use_heuristics:
                prelim_predictions_per_feature = sorted(
                    prelim_predictions_per_feature,
                    key=lambda x: (x.start_logit + x.end_logit - (x.end_index - x.start_index + 1)),
                    reverse=True)
            else:
                prelim_predictions_per_feature = sorted(
                    prelim_predictions_per_feature,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)

            for i, pred_i in enumerate(prelim_predictions_per_feature):
                if len(span_starts) >= int(n_best_size)/2:
                    break
                final_text = wrapped_get_final_text(example, feature, pred_i.start_index, pred_i.end_index,
                                                    do_lower_case, verbose_logging, logger)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True

                span_starts.append(pred_i.start_index)
                span_ends.append(pred_i.end_index)
                labels.append(0)
                label_masks.append(1)

                # filter out redundant candidates
                if (i+1) < len(prelim_predictions_per_feature) and use_nms:
                    indexes = []
                    for j, pred_j in enumerate(prelim_predictions_per_feature[(i+1):]):
                        filter_text = wrapped_get_final_text(example, feature, pred_j.start_index, pred_j.end_index,
                                                             do_lower_case, verbose_logging, logger)
                        if filter_type == 'em':
                            if exact_match_score(final_text, filter_text):
                                indexes.append(i + j + 1)
                        elif filter_type == 'f1':
                            if f1_score(final_text, filter_text) > 0:
                                indexes.append(i + j + 1)
                        else:
                            raise Exception
                    [prelim_predictions_per_feature.pop(index - k) for k, index in enumerate(indexes)]

        # Pad to fixed length
        while len(span_starts) < int(n_best_size):
            span_starts.append(0)
            span_ends.append(0)
            labels.append(0)
            label_masks.append(0)
        assert len(span_starts) == int(n_best_size)
        assert len(span_ends) == int(n_best_size)
        assert len(labels) == int(n_best_size)
        assert len(label_masks) == int(n_best_size)

        batch_span_starts.append(span_starts)
        batch_span_ends.append(span_ends)
        batch_labels.append(labels)
        batch_label_masks.append(label_masks)
    return batch_span_starts, batch_span_ends, batch_labels, batch_label_masks


def ts2start_end(ts_tag_sequence):
    starts, ends = [], []
    n_tag = len(ts_tag_sequence)
    prev_pos, prev_sentiment = '$$$', '$$$'
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag =='T-NEG-B' or cur_ts_tag == 'T-POS-B' or cur_ts_tag == 'T-NEU-B':
            starts.append(i)
            if prev_pos !='O' and prev_pos !='$$$':
                ends.append(i-1)
            prev_pos=cur_ts_tag
        elif cur_ts_tag =='O':
            if prev_pos !='O' and prev_pos !='$$$':
                ends.append(i-1)
            prev_pos=cur_ts_tag
        elif cur_ts_tag == 'T-NEG' or cur_ts_tag == 'T-POS' or cur_ts_tag == 'T-NEU':
            prev_pos = cur_ts_tag
        elif cur_ts_tag == 'B-X':
            if prev_pos!='O':
                ends.append(i - 1)
                break
        else:
            raise Exception('!! find error tag:{}'.format(cur_ts_tag))
        if prev_pos!='O' and i == n_tag - 1:
            ends.append(n_tag - 1)
    assert len(starts) == len(ends)
    return starts,ends

def ts2polarity(words, ts_tag_sequence, starts, ends):
    polarities = []
    for start, end in zip(starts, ends):
        cur_ts_tag = ts_tag_sequence[start]
        cur_pos, cur_sentiment, = cur_ts_tag.split('-')[:2]
        assert cur_pos == 'T'
        prev_sentiment = cur_sentiment
        if start < end:
            for idx in range(start, end + 1):
                cur_ts_tag = ts_tag_sequence[idx]
                cur_pos, cur_sentiment = cur_ts_tag.split('-')[:2]
                assert cur_pos == 'T'
                assert cur_sentiment == prev_sentiment, (words, ts_tag_sequence, start, end)
                prev_sentiment = cur_sentiment
        polarities.append(cur_sentiment)
    return polarities


def pos2term(words, starts, ends):
    term_texts = []
    for start, end in zip(starts, ends):
        term_texts.append(' '.join(words[start:end+1]))
    return term_texts

def image_process(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))])
        # transforms.Normalize((0.48, 0.498, 0.531),
        #                  (0.214, 0.207, 0.207))])

    image = Image.open(image_path).convert('RGB')
    # image  =image.resize((600,400))
    image = transform(image)
    return image


def convert_absa_data(img_path,dataset,args,verbose_logging=False,gcn_features=None,gcn_datasets=None):

    examples = []
    n_records = len(dataset)
    n = len(dataset['words'])
    count=0
    dep,adj,dep_text,adj_matrix,src_mask,aspect_mask,polaritys = None,None,None,None,None,None,None
    for i in range(n):
        words = dataset['words'][i] #[i]['words']
        # if SEP_TAG in words:
        #     words = replace_sep_token(words, args)
        ts_tags = dataset['ts_targets'][i]
        image_labels = dataset['image_labels'][i]
        image_ids = dataset['imgs'][i]

        starts, ends = ts2start_end(ts_tags)
        polarities = ts2polarity(words, ts_tags, starts, ends)
        term_texts = pos2term(words, starts, ends)
        base_image_path=img_path
        image_id = image_ids[0]
        image_path=base_image_path+'/'+image_id
        cache_dir = args.cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if args.dataset_name=='twitter15':  # twitter数据集
            cache_path=cache_dir+'tw15_img/'+image_id[:-4]+'.tch'
        elif args.dataset_name=='twitter17':
            cache_path=cache_dir+'tw17_img/'+image_id+'.tch'
        else:
            raise ValueError('image path error')
        if os.path.exists(cache_path):
            raw_image_data=torch.load(cache_path)
        else:
            try:
                raw_image_data = image_process(image_path)# tensor
            except:
                count += 1
                print(" Can not find image {}".format(image_path))
                img_path = os.path.join(raw_image_data, '17_06_4705.jpg')
                raw_image_data = image_process(image_path)# tensor
            torch.save(raw_image_data, cache_path)


        if term_texts != []:
            new_polarities = []
            for polarity in polarities:
                if polarity == 'POS':
                    new_polarities.append('positive')
                elif polarity == 'NEG':
                    new_polarities.append('negative')
                elif polarity == 'NEU':
                    new_polarities.append('neutral')
                else:
                    raise Exception
            assert len(term_texts) == len(starts)
            assert len(term_texts) == len(new_polarities)
            if gcn_features:
                # text_a = [x['word'] for x in gcn_features[i]]
                # label = [x['tag'] for x in gcn_features[i]]
                # label = ot2bieos_ts(label)  #BIOES
                dep = [x['dep'] for x in gcn_features[i]]
                adj = [x['adj'] for x in gcn_features[i]]
                dep_text = [x['dep_text'] for x in gcn_features[i]]
                example = SemEvalExample1(str(i), words, term_texts, starts, ends, new_polarities, image_labels,image_ids, raw_image_data, dep, adj, dep_text, adj_matrix)  # 后4gcn
            elif gcn_datasets:
                adj_matrix = gcn_datasets[i]['adj_matrix']
                src_mask = gcn_datasets[i]['src_mask']
                aspect_mask = gcn_datasets[i]['aspect_mask']
                polaritys = gcn_datasets[i]['polarity']
                example = SemEvalExample2(str(i), words, term_texts, starts, ends, new_polarities, image_labels, image_ids, raw_image_data, adj_matrix, src_mask, aspect_mask, polaritys)  # 后4gcn
            else:
                example = SemEvalExample(str(i), words, term_texts, starts, ends, new_polarities,image_labels,image_ids,raw_image_data)
            examples.append(example)
            if i < 50 and verbose_logging:
                print(example)
    print("Convert %s examples" % len(examples))
    return examples

def read_absa_data(path, sample_ratio=1.0):
    """
    read data from the specified path
    :param path: path of dataset
    :return: dict
    """
    dataset = []
    i=1
    sentences, words, ote_targets, ts_targets, labels, img_ids = [], [], [], [], [], []
    with open(path, encoding='UTF-8') as fp:
        for line in fp:
            record = {}
            sent, tag_string = line.strip().split('####')
            tag_string,img_string,image_ids_string=tag_string.strip().split('____')
            _,img_labels=img_string.split('=')
            labels.append([int(item.strip()) for item in  img_labels[1:-1].split(',')])
            _,img_ids_s=image_ids_string.split('=')
            img_ids.append([item.strip() for item in img_ids_s[2:-2].split(',')])
            record['sentence'] = sent
            sentences.append(sent)
            word_tag_pairs = tag_string.strip().split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # tag sequence for opinion target extraction
            ote_tags = []
            # word sequence
            wordlist = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                wordlist.append(word.lower())
                if tag == 'O':
                    ote_tags.append('O')
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ote_tags.append('T')
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU')
                elif tag == 'T-NEG-B':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG-B')
                elif tag == 'T-NEU-B':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU-B')
                elif tag == 'T-POS-B':
                    ote_tags.append('T')
                    ts_tags.append('T-POS-B')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            record['words'] = words.copy()
            words.append(wordlist)
            record['ote_raw_tags'] = ote_tags.copy()
            ote_targets.append(ote_tags)
            record['ts_raw_tags'] = ts_tags.copy()
            ts_targets.append(ts_tags)
            record['image_labels'] = labels.copy()
            record['image_ids'] = img_ids.copy()
            dataset.append(record)
            i+=1
    print("Obtain %s records from %s" % (len(dataset), path))

    assert len(sentences) == len(words) == len(ote_targets) == len(ts_targets)== len(labels) == len(img_ids)
    return {"sentences":sentences, "words": words, "ote_targets": ote_targets, "ts_targets": ts_targets, \
            "image_labels": labels, "imgs": img_ids}

def read_agn_data(path, dataset, mode):
    """
    read agn_data from the specified path
    :param path: path of dataset
    :return: dict
    """
    # for txt in data:
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()
        img_id = []
        for i in range(len(lines)):
            img_id.append(lines[i].split(" ")[0])

        for i in range(len(lines)):
            for j in range(len(dataset['imgs'])):
                if img_id[i] == "".join(dataset['imgs'][j]):
                    dataset['sentences'][j] = dataset['sentences'][j] + " ".join(lines[i].strip().split(".jpg")[1:])
                    dataset['words'][j].extend(lines[i].strip().split(" ")[1:])
                    dataset['ote_targets'][j].extend(['X'] * len(lines[i].strip().split(" ")[1:]))
                    dataset['ts_targets'][j].extend(['B-X'] * len(lines[i].strip().split(" ")[1:]))
                    assert len(dataset['words'][j]) == len(dataset['ote_targets'][j]) == len(dataset['ts_targets'][j])

    return {"sentences":dataset['sentences'], "words": dataset['words'], "ote_targets": dataset['ote_targets'], "ts_targets": dataset['ts_targets'],
            'image_labels': dataset['image_labels'], "imgs": dataset['imgs']}

def read_agn_data2(path, dataset, mode):
    """
    read agn_data from the specified path
    :param path: path of dataset
    :return: dict
    """
    # for txt in data:
    with open(path, 'r', encoding="utf8") as f:
        lines = f.readlines()
        img_id = []
        for i in range(len(lines)):
            img_id.append(lines[i].split(" ")[0])

        for i in range(len(lines)):
            for j in range(len(dataset['imgs'])):
                if img_id[i] == "".join(dataset['imgs'][j]):
                    dataset['sentences'][j] = dataset['sentences'][j] + " ".join(lines[i].strip().split(".jpg")[1:])
                    dataset['words'][j].extend(lines[i].strip().split(" ")[1:])
                    dataset['ote_targets'][j].extend(['X'] * len(lines[i].strip().split(" ")[1:]))
                    dataset['ts_targets'][j].extend(['B-X'] * len(lines[i].strip().split(" ")[1:]))
                    assert len(dataset['words'][j]) == len(dataset['ote_targets'][j]) == len(dataset['ts_targets'][j])

    return {"sentences":dataset['sentences'], "words": dataset['words'], "ote_targets": dataset['ote_targets'], "ts_targets": dataset['ts_targets'],
            'image_labels': dataset['image_labels'], "imgs": dataset['imgs']}


def to_tensor(array):
    return torch.from_numpy(np.array(array)).float()


def to_variable(tensor, requires_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=requires_grad)


def my_relu(data):
    data[data < 1e-8] = 1e-8
    return data


class E2EASAOTProcessor(object):
    def __init__(self, direct=False):
        self.direct = direct
        self.train_examples = None
        self.dev_examples = None
        self.test_examples = None

    def get_type_num(self):
        type_num = 100 if self.direct else 50
        return type_num

    def get_label_num(self):
        label_list = self.get_labels()
        return len(label_list) + 1

    @classmethod
    def _read_tsv(cls, input_file):
        '''
        read file
        return format :
        '''
        f = open(input_file)
        data = []
        sentence = []
        label = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue
            splits = line.strip().split('\t')
            sentence.append(splits[0])
            label.append(splits[-1])

        if len(sentence) > 0:
            data.append((sentence, label))
        return data

    def get_labels(self):
        return ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU', "[CLS]", "[SEP]"]

def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x