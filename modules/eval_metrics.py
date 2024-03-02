"""Run BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import json
import collections

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from squad.squad_evaluate import exact_match_score
from models.utils import read_absa_data, convert_absa_data, convert_examples_to_features, \
    RawFinalResult, wrapped_get_final_text, id_to_label, E2EASAOTProcessor

try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_train_data(img_path, transform, data_dict, max_seq, args, tokenizer, logger):
    train_examples = convert_absa_data(img_path, transform, dataset=data_dict, args=args,
                      verbose_logging=args.verbose_logging)  # transform the data into the example class

    train_features = convert_examples_to_features(train_examples, tokenizer, max_seq, args.verbose_logging, logger)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

    logger.info("Num orig examples = %d", len(train_examples))
    logger.info("Num split features = %d", len(train_features))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in train_features], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in train_features], dtype=torch.long)
    all_labels = torch.tensor([f.polarity_labels for f in train_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in train_features], dtype=torch.long)
    if args.bert_name.startswith("roberta"):
        train_data = TensorDataset(all_input_ids, all_input_mask, all_span_starts, all_span_ends,all_labels, all_label_masks)
    else:
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends, all_labels, all_label_masks)
    train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_examples, train_features, train_dataloader


def read_eval_data(img_path, transform, data_dict, max_seq, args, tokenizer, logger, mode="test"):
    eval_examples = convert_absa_data(img_path, transform, dataset=data_dict, args=args,
                             verbose_logging=args.verbose_logging)  # transform the data into the example class

    eval_features = convert_examples_to_features(eval_examples, tokenizer, max_seq, args.verbose_logging, logger)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    logger.info("Num orig examples = %d", len(eval_examples))
    logger.info("Num split features = %d", len(eval_features))
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_span_starts = torch.tensor([f.start_indexes for f in eval_features], dtype=torch.long)
    all_span_ends = torch.tensor([f.end_indexes for f in eval_features], dtype=torch.long)
    all_label_masks = torch.tensor([f.label_masks for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    if args.bert_name.startswith("roberta"):
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_span_starts, all_span_ends, all_label_masks, all_example_index)
    else:
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_span_starts, all_span_ends, all_label_masks, all_example_index)

    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return eval_examples, eval_features, eval_dataloader


def metric_max_over_ground_truths(metric_fn, term, polarity, gold_terms, gold_polarities):
    hit = 0
    for gold_term, gold_polarity in zip(gold_terms, gold_polarities):
        score = metric_fn(term, gold_term)
        if score and polarity == gold_polarity:
            hit = 1
    return hit


def eval_absa(all_examples, all_features, all_results, do_lower_case, verbose_logging, logger):
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    all_nbest_json = collections.OrderedDict()
    common, relevant, retrieved = 0., 0., 0.

    for (feature_index, feature) in enumerate(all_features):
        example = all_examples[feature.example_index]
        result = unique_id_to_result[feature.unique_id]

        pred_terms = []
        pred_polarities = []
        for start_index, end_index, cls_pred, span_mask in \
                zip(result.start_indexes, result.end_indexes, result.cls_pred, result.span_masks):
            if span_mask:
                final_text = wrapped_get_final_text(example, feature, start_index, end_index,
                                                    do_lower_case, verbose_logging, logger)
                pred_terms.append(final_text)
                pred_polarities.append(id_to_label[cls_pred])

        prediction = {'pred_terms': pred_terms, 'pred_polarities': pred_polarities, 'gold_terms': example.term_texts,
                      'gold_polarites': example.polarities}
        all_nbest_json[example.example_id] = prediction

        for term, polarity in zip(pred_terms, pred_polarities):
            common += metric_max_over_ground_truths(exact_match_score, term, polarity, example.term_texts,
                                                    example.polarities)

        retrieved += len(pred_terms)
        relevant += len(example.term_texts)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (2 * p * r) / (p + r) if p > 0 and r > 0 else 0.
    return {'p': p, 'r': r, 'f1': f1, 'common': common, 'retrieved': retrieved, 'relevant': relevant}, all_nbest_json

def eval_json(all_examples, all_features, y_true, y_pred):
    no_com = ['[SEP]', 'PAD', 'X', 'O', '[CLS]']
    all_nbest_json = collections.OrderedDict()
    error_nbest_json = collections.OrderedDict()
    assert len(all_examples) == len(all_features) == len(y_true) == len(y_pred)
    for index in range(len(y_true)):
        gold_terms = []
        gold_labels = []
        pred_terms = []
        pred_labels = []
        gold_term = ""
        gold_label = ""
        pred_term = ""
        pred_label = ""
        for (i, true_label) in enumerate(y_true[index]):
            if true_label not in no_com:
                gold_term+=''.join(all_examples[index].text_a.split()[i:i+1]) + " "
                gold_label+=''.join(true_label) + " "
            else:
                if gold_term or gold_label:
                    gold_terms.append(gold_term)
                    gold_labels.append(gold_label)
                    gold_term = ""
                    gold_label = ""
            if y_pred[index][i] not in no_com:
                pred_term+=''.join(all_examples[index].text_a.split()[i:i+1]) + " "
                pred_label+=''.join(y_pred[index][i]) + " "
            else:
                if pred_term or pred_label:
                    pred_terms.append(pred_term)
                    pred_labels.append(pred_label)
                    pred_term = ""
                    pred_label = ""
        if gold_term!="" or gold_label!="":
            gold_terms.append(gold_term)
            gold_labels.append(gold_label)
        if gold_term != "" or gold_label != "":
            pred_terms.append(pred_term)
            pred_labels.append(pred_label)
        prediction = {'pred_terms': pred_terms, 'pred_labels': pred_labels, 'gold_terms': gold_terms,'gold_labels': gold_labels}
        if set(pred_terms) != set(gold_terms):
            error = {'pred_terms': pred_terms, 'pred_labels': pred_labels, 'gold_terms': gold_terms,'gold_labels': gold_labels}
            error_nbest_json[all_examples[index].guid] = error
        all_nbest_json[all_examples[index].guid] = prediction
    return all_nbest_json,error_nbest_json

def eval(args, model, device, eval_examples, eval_features, eval_dataloader, logger, write_pred=False):
    all_results = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, span_starts, span_ends, label_masks, example_indices = batch
        with torch.no_grad():
            cls_logits = model('inference', input_mask, input_ids=input_ids, token_type_ids=segment_ids,
                               span_starts=span_starts, span_ends=span_ends)

        for j, example_index in enumerate(example_indices):
            cls_pred = cls_logits[j].detach().cpu().numpy().argmax(axis=1).tolist()
            start_indexes = span_starts[j].detach().cpu().tolist()
            end_indexes = span_ends[j].detach().cpu().tolist()
            span_masks = label_masks[j].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                              end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

    metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                        args.do_lower_case, args.verbose_logging, logger)

    if write_pred:
        output_file = os.path.join(args.output_dir, "predictions.json")
        with open(output_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        logger.info("Writing predictions to: %s" % (output_file))
    return metrics

def get_chunks(seq, tags):
	default = tags['O']
	idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq))
		chunks.append(chunk)

	return chunks

def get_chunk_type(tok, idx_to_tag):
	tag_name = idx_to_tag[tok]
	tag_class = tag_name.split('-')[0]
	tag_type = tag_name.split('-')[-1]
	return tag_class, tag_type

def evaluate(labels_pred, labels, tags):
    accs = []
    correct_preds, total_correct, total_preds = 0., 0., 0.

    for lab, lab_pred in zip(labels, labels_pred):
        lab = lab
        lab_pred = lab_pred
        accs += [a == b for (a, b) in zip(lab, lab_pred)]

        lab_chunks = set(get_chunks(lab, tags))
        lab_pred_chunks = set(get_chunks(lab_pred, tags))
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    acc = np.mean(accs)

    return acc, f1, p, r


def evaluate_each_class(labels_pred, labels, tags, class_type):
    correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

    for lab, lab_pred in zip(labels, labels_pred):
        lab_pre_class_type = []
        lab_class_type = []

        lab = lab
        lab_pred = lab_pred
        lab_chunks = get_chunks(lab, tags)
        lab_pred_chunks = get_chunks(lab_pred, tags)
        for i in range(len(lab_pred_chunks)):
            if lab_pred_chunks[i][0] == class_type:
                lab_pre_class_type.append(lab_pred_chunks[i])
        lab_pre_class_type_c = set(lab_pre_class_type)

        for i in range(len(lab_chunks)):
            if lab_chunks[i][0] == class_type:
                lab_class_type.append(lab_chunks[i])
        lab_class_type_c = set(lab_class_type)

        lab_chunksss = set(lab_chunks)
        correct_preds_cla_type += len(lab_pre_class_type_c & lab_chunksss)
        total_preds_cla_type += len(lab_pre_class_type_c)
        total_correct_cla_type += len(lab_class_type_c)

    p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
    r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0

    return f1, p, r


def eval_result(true_labels, pred_result, rel2id, logger, use_name=False):
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in rel2id:
            if use_name:
                neg = name
            else:
                neg = rel2id[name]
            break
    for i in range(total):
        if use_name:
            golden = true_labels[i]
        else:
            golden = true_labels[i]

        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result

def eval_asa(true_labels, pred_result, lab2id, logger, use_name=False):
    label_to_id = {'other': 0, 'neutral': 1, 'positive': 2, 'negative': 3, 'conflict': 4}
    lab2id = label_to_id
    correct = 0
    total = len(true_labels)
    correct_positive = 0
    pred_positive = 0
    gold_positive = 0

    neg = -1
    for name in ['other', 'neutral', 'positive', 'negative', 'conflict']:
        if name in lab2id:
            if use_name:
                neg = name
            else:
                neg = lab2id[name]
            break
    for i in range(total):
        if use_name:
            golden = true_labels[i]
        else:
            golden = true_labels[i]

        if golden == pred_result[i]:
            correct += 1
            if golden != neg:
                correct_positive += 1
        if golden != neg:
            gold_positive += 1
        if pred_result[i] != neg:
            pred_positive += 1
    acc = float(correct) / float(total)
    try:
        micro_p = float(correct_positive) / float(pred_positive)
    except:
        micro_p = 0
    try:
        micro_r = float(correct_positive) / float(gold_positive)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    logger.info('Evaluation result: {}.'.format(result))
    return result