import json
import logging

import os
import torch
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers import BertConfig
from transformers.optimization import get_linear_schedule_with_warmup

from models import TVNetSAModel
from models.utils import span_annotate_candidates, RawSpanResult, convert_absa_data, convert_examples_to_features, \
    label_to_id, RawFinalResult
from modules.augument import Cutoff

from .eval_metrics import read_eval_data, read_train_data, eval_absa, evaluate, evaluate_each_class, eval_json
from models.modeling_bert import BertModel

import torch.nn.functional as F
logger = logging.getLogger(__name__)
class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

class SATrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, label_map=None,
                 args=None, logger=None, writer=None, train_dataset=None, dev_dataset=None, test_dataset=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.label_map = label_map #label
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        if self.train_data is not None:
            self.train_num_steps = int(len(self.train_data) / args.gradient_accumulation_steps) * args.num_epochs
            if args.local_rank != -1:
                self.train_num_steps = self.train_num_steps // torch.distributed.get_world_size()
        self.step = 0
        self.args = args

    def train(self):
        if self.args.use_prefix:
            self.multiModal_before_train()
        else:
            self.bert_before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.train_batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Train Batch size = %d", self.args.train_batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        global_step = 0

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            if self.args.use_pretrained:
                self.load_pretrained()
            else:
                self.model.load_state_dict(torch.load(self.args.load_path,map_location=torch.device('cpu')))
            self.logger.info("Load model successful!")

        examples, features, img_path, data_dict, tokenizer, max_seq = self.train_dataset.examples, self.train_dataset.features, self.train_dataset.img_path, self.train_dataset.data_dict, self.train_dataset.tokenizer, self.train_dataset.max_seq
        all_results = []
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss,avg_prob_loss,avg_ori_loss = 0,0,0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    if self.args.use_probe:
                        attention_mask, labels, batch_ac_logits, loss, span_starts, span_ends,\
                        label_masks, train_examples, train_features, example_indices, prob_loss, ori_loss = self._step(batch, examples, features, mode="train")  #得到loss和logits

                        prob_loss = prob_loss / self.args.gradient_accumulation_steps
                        ori_loss = ori_loss.detach().cpu().item()
                        avg_prob_loss += prob_loss.detach().cpu().item()
                        avg_ori_loss += ori_loss
                    else:
                        attention_mask, labels, batch_ac_logits, loss, span_starts, span_ends, \
                        label_masks, train_examples, train_features, example_indices = self._step(batch, examples, features, mode="train")  # 得到loss和logits

                    loss = loss / self.args.gradient_accumulation_steps  # 损失标准化（可选，如果损失要在训练样本上取平均）

                    avg_loss += loss.detach().cpu().item()

                    loss.backward()                                       # 反向传播，计算梯度
                    if (self.step + 1) % self.args.gradient_accumulation_steps == 0:

                        self.optimizer.step()                             # 更新参数
                        self.scheduler.step()                             # 热启动，也相关
                        self.optimizer.zero_grad()                        # 梯度清零
                        global_step += 1
                    for j, example_index in enumerate(example_indices):
                        cls_pred = batch_ac_logits[j].argmax(axis=1).tolist()
                        start_indexes = span_starts[j].detach().cpu().tolist()
                        end_indexes = span_ends[j].detach().cpu().tolist()
                        span_masks = label_masks[j]
                        train_feature = train_features[example_index.item()]
                        unique_id = int(train_feature.unique_id)
                        all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                                          end_indexes=end_indexes, cls_pred=cls_pred, span_masks=span_masks))

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        avg_prob_loss = float(avg_prob_loss) / self.refresh_step
                        avg_ori_loss = float(avg_ori_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f},prob_loss:{:<6.5f},ori_loss:{:<6.5f}".format(avg_loss,avg_prob_loss,avg_ori_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                   global_step=self.step)
                        avg_loss,avg_prob_loss,avg_ori_loss = 0,0,0

                metrics, all_nbest_json =eval_absa(train_examples, train_features, all_results,
                                                    self.args.do_lower_case, self.args.verbose_logging, logger)

                self.logger.info("=======================================================")
                self.logger.info("***** Train Eval results *****")
                f1_score = metrics['f1']
                if self.writer:
                    self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)
                self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}, P: {:.4f}, R: {:.4f}." \
                                 .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch,
                                         f1_score, metrics['p'], metrics['r']))
                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch
                
                write_pred = True
                if write_pred:
                    output_file = os.path.join(self.args.save_path, "train_predictions.json")
                    with open(output_file, "w") as writer:
                        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
                    logger.info("Writing predictions to: %s" % (output_file))

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)  # generator to dev.

            torch.cuda.empty_cache()

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.eval_batch_size)
        self.logger.info("  Eval Batch size = %d", self.args.eval_batch_size)

        y_true, y_pred = [], []
        all_results = []
        step = 0
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0

                for batch in self.dev_data:
                    step += 1

                    attention_mask, labels, batch_ac_logits, loss ,span_starts, span_ends, \
                    label_masks, eval_examples, eval_features, example_indices = self._step(batch, self.dev_dataset.examples, self.dev_dataset.features,mode="dev")
                    total_loss += loss.detach().cpu().item()

                    for j, example_index in enumerate(example_indices):
                        cls_pred = batch_ac_logits[j].argmax(axis=1).tolist()
                        start_indexes = span_starts[j].detach().cpu().tolist()
                        end_indexes = span_ends[j].detach().cpu().tolist()
                        span_masks = label_masks[j]
                        eval_feature = eval_features[example_index.item()]
                        unique_id = int(eval_feature.unique_id)
                        all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                                          end_indexes=end_indexes, cls_pred=cls_pred,
                                                          span_masks=span_masks))

                    pbar.update()
                metrics, all_nbest_json = eval_absa(eval_examples, eval_features, all_results,
                                                    self.args.do_lower_case, self.args.verbose_logging, logger)
                pbar.close()


                for i in range(len(y_true)):
                    for j in range(len(y_true[i])):
                        y_true[i][j] = y_true[i][j][0] + y_true[i][j]
                        y_pred[i][j] = y_pred[i][j][0] + y_pred[i][j]

                self.logger.info("=======================================================")
                self.logger.info("***** Dev Eval results *****")

                f1_score = metrics['f1']
                if self.writer:
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / step,
                                           global_step=epoch)  # tensorbordx

                self.logger.info(
                    "Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {:.4f},loss: {:.4f}, P: {:.4f}, R: {:.4f}." \
                    .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                            f1_score, total_loss / step, metrics['p'], metrics['r']))
                if f1_score >= self.best_dev_metric:
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score  # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

                write_pred = True
                if write_pred:
                    output_file = os.path.join(self.args.save_path, "dev_predictions.json")
                    with open(output_file, "w") as writer:
                        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
                    logger.info("Writing predictions to: %s" % (output_file))

        self.model.train()

    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.eval_batch_size)
        self.logger.info("  Eval Batch size = %d", self.args.eval_batch_size)

        load_path = self.args.save_path + '/' + 'best_model.pth'
        if self.args.save_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(load_path))
            self.model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
            self.logger.info("Load model successful!")

        y_true, y_pred = [], []
        all_results = []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0

                for batch in self.test_data:
                    attention_mask, labels, batch_ac_logits, loss,  span_starts, span_ends, \
                    label_masks, test_examples, test_features, example_indices = self._step(batch, self.test_dataset.examples, self.test_dataset.features,mode="test")  # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    for j, example_index in enumerate(example_indices):
                        cls_pred = batch_ac_logits[j].argmax(axis=1).tolist()
                        start_indexes = span_starts[j].detach().cpu().tolist()
                        end_indexes = span_ends[j].detach().cpu().tolist()
                        span_masks = label_masks[j]
                        test_feature = test_features[example_index.item()]
                        unique_id = int(test_feature.unique_id)
                        all_results.append(RawFinalResult(unique_id=unique_id, start_indexes=start_indexes,
                                                          end_indexes=end_indexes, cls_pred=cls_pred,
                                                          span_masks=span_masks))

                    pbar.update()

                metrics, all_nbest_json = eval_absa(test_examples, test_features, all_results,
                                                    self.args.do_lower_case, self.args.verbose_logging, logger)
                pbar.close()

                for i in range(len(y_true)):
                    for j in range(len(y_true[i])):
                        y_true[i][j] = y_true[i][j][0] + y_true[i][j]
                        y_pred[i][j] = y_pred[i][j][0] + y_pred[i][j]

                self.logger.info("=======================================================")
                self.logger.info("***** Test Eval results *****")

                f1_score = metrics['f1']
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score)
                    self.writer.add_scalar(tag='test_loss',
                                           scalar_value=total_loss / len(self.test_data))

                self.logger.info("Test f1 score: {}, P: {:.4f}, R: {:.4f}.".format(f1_score, metrics['p'], metrics['r']))
                write_pred = True
                if write_pred:
                    output_file = os.path.join(self.args.save_path, "test_predictions.json")
                    with open(output_file, "w") as writer:
                        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
                    logger.info("Writing predictions to: %s" % (output_file))

        self.model.train()
        return f1_score

    def _step(self, batch, examples, features, mode="train"):
        device = self.args.device
        if self.args.device != 'cpu':
            batch = tuple(t.to(self.args.device) for t in batch)
            # batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
        if self.args.use_prefix:
            if self.args.gcn_layer_number > 0:
                input_ids, attention_mask, segment_ids, example_indices, start_positions, end_positions, \
                bio_labels, polarity_positions, images, aux_imgs , \
                valid_ids, b_use_valid_filter, adj_matrix, dep_matrix = batch
                valid_ids=valid_ids.to(device)
                adj_matrix=adj_matrix.to(device)
            elif self.args.num_layers > 0:
                input_ids, attention_mask, segment_ids, example_indices, start_positions, end_positions, bio_labels, polarity_positions, images, aux_imgs , \
                adj_matrix, src_mask, aspect_mask, polaritys = batch
                adj_matrix = adj_matrix.to(device)
                src_mask = src_mask.to(device)
                aspect_mask = aspect_mask.to(device)
                polaritys = polaritys.to(device)
            else:
                input_ids,  attention_mask, segment_ids, example_indices, start_positions, end_positions,\
                bio_labels, polarity_positions, images, aux_imgs = batch
            bsz = input_ids.size(0)
            prefix_guids = self.model.get_visual_prompt(images, aux_imgs)
            prefix_guids_length = prefix_guids[0][0].shape[2]
            prefix_guids_mask = torch.ones((bsz, prefix_guids_length)).to(self.args.device)
            prompt_attention_mask = torch.cat((prefix_guids_mask, attention_mask), dim=1)

        else:
            images, aux_imgs = None, None
            if self.args.gcn_layer_number > 0:
                input_ids, attention_mask, segment_ids, example_indices, start_positions, end_positions, \
                bio_labels, polarity_positions, valid_ids, b_use_valid_filter, adj_matrix, dep_matrix = batch
            elif self.args.num_layers > 0:
                input_ids, attention_mask, segment_ids, example_indices, start_positions, end_positions, \
                bio_labels, polarity_positions, adj_matrix, src_mask, aspect_mask, polaritys = batch
            else:
                input_ids, attention_mask, segment_ids, example_indices, start_positions, end_positions, \
                bio_labels, polarity_positions = batch
            prefix_guids = None
            prompt_attention_mask = attention_mask

        if self.args.use_probe:
            if mode == "train" and self.args.do_aug:
                batch_start_logits, batch_end_logits, _, _ = self.model.extraction(prompt_attention_mask, input_ids, prefix_guids, segment_ids, True)
            if mode=="train":
                batch_start_logits, batch_end_logits, _, _ = self.model.extraction(prompt_attention_mask, input_ids, prefix_guids, segment_ids)
            else:
                with torch.no_grad():
                    batch_start_logits, batch_end_logits, _, _ = self.model.extraction(prompt_attention_mask, input_ids, prefix_guids, segment_ids)
        elif self.args.num_layers > 0:
            if mode == "train":
                gcn_logits, penal, batch_start_logits, batch_end_logits, _ = self.model.extraction(prompt_attention_mask, input_ids, prefix_guids, segment_ids, adj_matrix=adj_matrix, src_mask=src_mask, aspect_mask=aspect_mask)
            else:
                with torch.no_grad():
                    gcn_logits, penal, batch_start_logits, batch_end_logits, _ = self.model.extraction(prompt_attention_mask, input_ids, prefix_guids, segment_ids, adj_matrix=adj_matrix, src_mask=src_mask, aspect_mask=aspect_mask)
        elif mode=="train" and self.args.do_aug:
            batch_start_logits, batch_end_logits, _ = self.model.extraction(prompt_attention_mask,input_ids, prefix_guids,segment_ids, True)
        elif mode=="train":
            batch_start_logits, batch_end_logits, _ = self.model.extraction(prompt_attention_mask, input_ids, prefix_guids, segment_ids)
        else:
            with torch.no_grad():
                batch_start_logits, batch_end_logits, _ = self.model.extraction(prompt_attention_mask,input_ids, prefix_guids,segment_ids)

        batch_features, batch_results = [], []

        for j, example_index in  enumerate(example_indices):
            start_logits = batch_start_logits[j].detach().cpu().tolist()
            end_logits = batch_end_logits[j].detach().cpu().tolist()
            feature = features[example_index.item()]
            unique_id = int(feature.unique_id)
            batch_features.append(feature)
            batch_results.append(RawSpanResult(unique_id = unique_id, start_logits = start_logits, end_logits = end_logits))

        span_starts, span_ends, labels, label_masks = span_annotate_candidates(examples, batch_features,
                                                                               batch_results,
                                                                               self.args.filter_type, mode,
                                                                               self.args.use_heuristics,
                                                                               self.args.use_nms,
                                                                               self.args.logit_threshold,
                                                                               self.args.n_best_size,
                                                                               self.args.max_answer_length,
                                                                               self.args.do_lower_case,
                                                                               self.args.verbose_logging, logger)

        span_starts = torch.tensor(span_starts, dtype=torch.long)
        span_ends = torch.tensor(span_ends, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        label_masks = torch.tensor(label_masks, dtype=torch.long)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        labels = labels.to(device)
        label_masks = label_masks.to(device)

        if self.args.use_prefix:
            if self.args.gcn_layer_number > 0:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions, polarity_labels=labels, \
                                    span_starts=span_starts, span_ends=span_ends, label_masks=label_masks, images=images, aux_imgs=aux_imgs, valid_ids=valid_ids, adjacency_matrix=adj_matrix)
            elif self.args.num_layers > 0:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks,images=images, aux_imgs=aux_imgs, adj_matrix=adj_matrix, src_mask=src_mask, aspect_mask=aspect_mask, polaritys=polaritys)
            else:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, \
                                    span_starts=span_starts, span_ends=span_ends,label_masks=label_masks, images=images, aux_imgs=aux_imgs)
        else:
            if self.args.gcn_layer_number > 0:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions, polarity_labels=labels, \
                                    span_starts=span_starts, span_ends=span_ends, label_masks=label_masks, valid_ids=valid_ids, adjacency_matrix=adj_matrix)
            elif self.args.num_layers > 0:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks, adj_matrix=adj_matrix, src_mask=src_mask, aspect_mask=aspect_mask, polaritys=polaritys)
            else:
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, start_positions=start_positions, end_positions=end_positions, polarity_labels=labels, \
                                    span_starts=span_starts, span_ends=span_ends, label_masks=label_masks)
        if self.args.use_probe:
            TokenClassifierOutput, prob_loss, ori_loss = output
            loss, logits = TokenClassifierOutput.loss, TokenClassifierOutput.logits
        else:
            logits, loss = output.logits, output.loss
        if mode == "train" and self.args.do_aug:
            if self.args.use_prefix:
                if self.args.gcn_layer_number > 0:
                    cutoff_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks,images=images, aux_imgs=aux_imgs, valid_ids=valid_ids,adjacency_matrix=adj_matrix, augument=True)
                elif self.args.num_layers > 0:
                    cutoff_output = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=segment_ids, start_positions=start_positions,end_positions=end_positions, polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks, images=images, aux_imgs=aux_imgs, augument=True, adj_matrix=adj_matrix, src_mask=src_mask, aspect_mask=aspect_mask)
                else:
                    cutoff_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks,images=images, aux_imgs=aux_imgs, augument=True)
            else:
                if self.args.gcn_layer_number > 0:
                    cutoff_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks, valid_ids=valid_ids, adjacency_matrix=adj_matrix, augument=True)
                elif self.args.num_layers > 0:
                    cutoff_output = self.model(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=segment_ids, start_positions=start_positions,end_positions=end_positions, polarity_labels=labels,span_starts=span_starts, span_ends=span_ends, label_masks=label_masks,augument=True, adj_matrix=adj_matrix, src_mask=src_mask, aspect_mask=aspect_mask, polaritys=polaritys)
                else:
                    cutoff_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids,start_positions=start_positions, end_positions=end_positions,polarity_labels=labels, span_starts=span_starts, span_ends=span_ends, label_masks=label_masks, augument=True)
            if self.args.use_probe:
                TokenClassifierOutput, prob_loss, ori_loss = cutoff_output
                cutoff_loss, cutoff_logits = TokenClassifierOutput.loss, TokenClassifierOutput.logits
            else:
                cutoff_logits, cutoff_loss = cutoff_output.logits, cutoff_output.loss
            loss = self.cal_cut_loss(loss, logits, cutoff_loss, cutoff_logits)
        if mode=="train" and self.args.use_probe:
            return attention_mask, labels, logits, loss, span_starts, span_ends, label_masks, examples, features, example_indices, prob_loss, ori_loss
        else:
            return attention_mask, labels, logits, loss, span_starts, span_ends, label_masks, examples, features, example_indices

    def bert_before_train(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)

    def multiModal_before_train(self):
        # bert lr
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        parameters.append(params)

        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        parameters.append(params)

        for name, par in self.model.named_parameters():  # freeze resnet
            if 'image_model' in name:   par.requires_grad = False

        self.optimizer = optim.AdamW(parameters, lr=self.args.lr)

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def load_pretrained(self):
        dict_trained = torch.load(self.args.load_path, map_location=torch.device('cpu'))
        dict_new = self.model.state_dict().copy()
        if False:
            dict_trained = {k: v for k, v in dict_trained.items() if k in dict_new}
            dict_new.update(dict_trained)
        else:
            trained_list = list(dict_trained.keys())
            new_list = list(dict_new.keys())
            j = 0
            no_load = {'dense', 'unary_affine', 'binary_affine', 'classifier'}
            for i in range(len(trained_list)):
                flag = False
                if 'crf' in trained_list[i]:
                    continue
                for nd in no_load:
                    if nd in new_list[j] and 'bert' not in new_list[j]:
                        flag = True
                if flag:
                    j += 8
                else:
                    if dict_new[new_list[j]].shape != dict_trained[trained_list[i]].shape:
                        j+=1
                        continue
                    dict_new[new_list[j]] = dict_trained[trained_list[i]]
                j += 1
        self.model.load_state_dict(dict_new)

    def cal_cut_loss(self, loss, logits, cutoff_loss, cutoff_logits):
        if self.args.aug_ce_loss > 0:
            loss += self.args.aug_ce_loss * cutoff_loss
        if self.args.aug_js_loss > 0:
            p = torch.softmax(logits + 1e-10, dim=1)
            q = torch.softmax(cutoff_logits + 1e-10, dim=1)
            aug_js_loss = self.js_div(p, q)  # JS离散一致性
            loss += self.args.aug_js_loss * aug_js_loss
        return loss

    def js_div(self, p, q):
        m = (p + q) / 2
        a = F.kl_div(p.log(), m, reduction='batchmean')
        b = F.kl_div(q.log(), m, reduction='batchmean')
        jsd = ((a + b) / 2)
        return jsd


class SATrainer2(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, label_map=None,
                 args=None, logger=None, writer=None, train_dataset=None, dev_dataset=None, test_dataset=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.label_map = label_map  # label
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None

        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        if self.train_data is not None:
            self.train_num_steps = int(len(
                self.train_data) / args.gradient_accumulation_steps) * args.num_epochs
            if args.local_rank != -1:
                self.train_num_steps = self.train_num_steps // torch.distributed.get_world_size()
        self.step = 0
        self.args = args

    def train(self):
        if self.args.use_prefix:
            self.multiModal_before_train()
        else:
            self.bert_before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.train_batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Train Batch size = %d", self.args.train_batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        global_step = 0
        if self.args.load_path is not None:
            self.logger.info("Loading model from {}".format(self.args.load_path))
            if self.args.use_pretrained:
                if self.args.use_152 or self.args.use_101 or self.args.use_34 or self.args.use_18:
                    self.load_pretrained2()
                else:
                    self.load_pretrained()
            else:
                self.model.load_state_dict(torch.load(self.args.load_path,map_location=torch.device('cpu')))
            self.logger.info("Load model successful!")
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss, avg_prob_loss, avg_img_loss = 0, 0, 0
            for epoch in range(1, self.args.num_epochs + 1):
                y_true, y_pred = [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    if self.args.use_probe:
                        attention_mask, labels, logits, loss, prob_loss, img_loss = self._step(batch, mode="train")
                        prob_loss = prob_loss / self.args.gradient_accumulation_steps
                        img_loss = img_loss.detach().cpu().item() if img_loss!=0 else img_loss
                        avg_prob_loss += prob_loss.detach().cpu().item()
                        avg_img_loss += img_loss
                    else:
                        attention_mask, labels, logits, loss = self._step(batch,  mode="train")

                    loss = loss / self.args.gradient_accumulation_steps

                    avg_loss += loss.detach().cpu().item()

                    loss.backward()  # 反向传播，计算梯度
                    if (self.step + 1) % self.args.gradient_accumulation_steps == 0:
                        self.optimizer.step()  # 更新参数
                        self.scheduler.step()  # 热启动，也相关
                        self.optimizer.zero_grad()  # 梯度清零
                        global_step += 1

                    label_ids = labels.to('cpu').numpy()
                    input_mask = attention_mask.to('cpu').numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    label_map[0] = "PAD"

                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []

                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])

                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        avg_prob_loss = float(avg_prob_loss) / self.refresh_step
                        avg_img_loss = float(avg_img_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f},prob_loss:{:<6.5f},img_tag_loss:{:<6.5f}".format(avg_loss,
                                                                                                   avg_prob_loss,
                                                                                                   avg_img_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                   global_step=self.step)
                        avg_loss, avg_prob_loss, avg_img_loss = 0, 0, 0


                self.logger.info("=======================================================")
                self.logger.info("***** Train Eval results *****")

                results = classification_report(y_true, y_pred, digits=4)
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])

                if self.writer:
                    self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)
                # , P: {:.4f}, R: {:.4f}.
                self.logger.info(
                    "Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}" \
                    .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch,f1_score))
                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)  # generator to dev.
                    print("进入test")
                    self.test(epoch)  # best test

            torch.cuda.empty_cache()

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.eval_batch_size)
        self.logger.info("  Eval Batch size = %d", self.args.eval_batch_size)
        y_true, y_pred = [], []
        step = 0
        dev_examples, dev_features = self.dev_dataset.examples, self.dev_dataset.features
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1

                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")
                    total_loss += loss.detach().cpu().item()

                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    label_map[0] = "PAD"
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    pbar.update()
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4)
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])

                if self.writer:
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / step,  global_step=epoch)

                self.logger.info(
                    "Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {:.4f},loss: {:.4f}." \
                        .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                f1_score, total_loss / step))
                if f1_score >= self.best_dev_metric:
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score  # update best metric(f1 score)
                    if not os.path.exists(self.args.save_path):
                        os.makedirs(self.args.save_path)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path + "/best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))
                    all_nbest_json,error_nbest_json = eval_json(dev_examples, dev_features, y_true, y_pred)

                    write_pred = True
                    if write_pred:
                        output_file = os.path.join(self.args.save_path, "dev_predictions.json")
                        with open(output_file, "w") as writer:
                            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
                        logger.info("Writing predictions to: %s" % (output_file))
                        output_file2 = os.path.join(self.args.save_path, "error_dev.json")
                        with open(output_file2, "w") as writer:
                            writer.write(json.dumps(error_nbest_json, indent=4) + "\n")
                        logger.info("Writing error test to: %s" % (output_file2))
        self.model.train()

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.eval_batch_size)
        self.logger.info("  Eval Batch size = %d", self.args.eval_batch_size)
        if(epoch == self.args.num_epochs):
            load_path = self.args.save_path + '/' + 'best_model.pth'
            if self.args.save_path is not None:  # load model from load_path
                self.logger.info("Test Loading model from {}".format(load_path))
                self.model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
                self.logger.info("Load model successful!")

        y_true, y_pred, y_true_idx, y_pred_idx = [], [], [], []
        test_examples, test_features = self.test_dataset.examples, self.test_dataset.features
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss= self._step(batch,mode="test")
                    total_loss += loss.detach().cpu().item()

                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    label_map[0] = "PAD"
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        true_label_idx = []
                        true_predict_idx = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[ label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                                    true_label_idx.append(label_ids[row][column])
                                    true_predict_idx.append(logits[row][column])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)
                        y_true_idx.append(true_label_idx)
                        y_pred_idx.append(true_predict_idx)

                    pbar.update()
                pbar.close()

                self.logger.info("***** Test Eval results *****")

                results = classification_report(y_true, y_pred, digits=4)
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score)
                    self.writer.add_scalar(tag='test_loss',
                                           scalar_value=total_loss / len(self.test_data))  # tensorbordx

                self.logger.info(
                    "Test f1 score: {}.".format(f1_score))

                self.logger.info(
                    "Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {:.4f}" \
                        .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch, f1_score))
                if f1_score >= self.best_test_metric:
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_test_epoch = epoch
                    self.best_test_metric = f1_score  # update best metric(f1 score)

                    all_nbest_json, error_nbest_json = eval_json(test_examples, test_features, y_true, y_pred)

                    write_pred = True
                    if write_pred:
                        output_file = os.path.join(self.args.save_path, "test_predictions.json")
                        with open(output_file, "w") as writer:
                            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
                        logger.info("Writing predictions to: %s" % (output_file))
                        output_file2 = os.path.join(self.args.save_path, "error_test.json")
                        with open(output_file2, "w") as writer:
                            writer.write(json.dumps(error_nbest_json, indent=4) + "\n")
                        logger.info("Writing error test to: %s" % (output_file2))

        self.model.train()
        return f1_score

    def _step(self, batch, mode="train"):
        device = self.args.device

        if self.args.device != 'cpu':
            batch = tuple(t.to(self.args.device) for t in batch)
            # batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
        if self.args.use_prefix:
            input_ids, attention_mask, segment_ids, labels, auxlabels, imagelabel, images, aux_imgs = batch#auxlabels没用
        else:
            auxlabels, imagelabel, images, aux_imgs = None, None, None, None
            input_ids, attention_mask, segment_ids, labels = batch

        labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(device)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids, \
                            labels=labels, imagelabel=imagelabel, images=images, aux_imgs=aux_imgs)

        if self.args.use_probe:
            TokenClassifierOutput, prob_loss, img_tag_loss = output
            loss, logits = TokenClassifierOutput.loss, TokenClassifierOutput.logits
        else:
            logits, loss = output.logits, output.loss
        if mode == "train" and self.args.use_probe:
            return attention_mask, labels, logits, loss, prob_loss, img_tag_loss
        else:
            return attention_mask, labels, logits, loss

    def bert_before_train(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)

    def multiModal_before_train(self):
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []

        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        parameters.append(params)

        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        parameters.append(params)

        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'crf' in name or name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters)

        for name, par in self.model.named_parameters():  # freeze resnet
            if 'image_model' in name:   par.requires_grad = False

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def load_pretrained(self):
        dict_trained = torch.load(self.args.load_path, map_location=torch.device('cpu'))
        dict_new = self.model.state_dict().copy()
        if False:
            dict_trained = {k: v for k, v in dict_trained.items() if k in dict_new}
            dict_new.update(dict_trained)
        else:
            trained_list = list(dict_trained.keys())
            new_list = list(dict_new.keys())
            j = 0
            no_load = {'dense', 'unary_affine', 'binary_affine', 'classifier'}
            for i in range(len(trained_list)):
                flag = False
                if 'crf' in trained_list[i]:
                    continue
                for nd in no_load:
                    if nd in new_list[j] and 'bert' not in new_list[j]:
                        flag = True
                if flag:
                    j += 8
                else:
                    if dict_new[new_list[j]].shape != dict_trained[trained_list[i]].shape:
                        j += 1
                        continue
                    dict_new[new_list[j]] = dict_trained[trained_list[i]]
                j += 1

        self.model.load_state_dict(dict_new, strict=False)

    def load_pretrained2(self):
        dict_trained = torch.load(self.args.load_path, map_location=torch.device('cpu'))
        dict_new = self.model.state_dict().copy()

        trained_list = list(dict_trained.keys())
        new_list = list(dict_new.keys())

        j = 0
        for i in range(len(trained_list)):
            if (i >= len(trained_list) or j >= len(new_list)):
                break
            if 'bert' in trained_list[i]:
                if dict_new[new_list[j]].shape != dict_trained[trained_list[i]].shape: # 不匹配
                    j += 1
                    continue
                dict_new[new_list[j]] = dict_trained[trained_list[i]]
            j += 1
        self.model.load_state_dict(dict_new, strict=False)

    def load_bert(self):
        dict_trained = torch.load(self.args.load_path, map_location=torch.device('cpu'))
        new_state_dict = self.model.state_dict()
        miss_keys = []
        for k in new_state_dict.keys():
            if k in dict_trained.keys():
                new_state_dict[k] = dict_trained[k]
            else:
                miss_keys.append(k)
        if len(miss_keys) > 0:
            logger.info('miss keys: {}'.format(miss_keys))
        self.model.load_state_dict(new_state_dict, strict=False)


