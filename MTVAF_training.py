import os
import argparse
import logging
import sys

from models.utils import E2EASAOTProcessor

sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from models.bert_model import TVNetSAModel, TVNetSAModel2
from modules.dataset import TVSAProcessor, TVSADataset, TVSAProcessor2, TVSADataset2
from modules.train import SATrainer, SATrainer2

import itertools
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL_CLASSES,TRAINER_CLASSES,DATA_PROCESS,DATA_PATH
MODEL_CLASSES = {
    'twitter15': TVNetSAModel,
    'twitter17': TVNetSAModel,
    'twitter2015': TVNetSAModel2,
    'twitter2017': TVNetSAModel2,
}

TRAINER_CLASSES = {
    'twitter15': SATrainer,
    'twitter17': SATrainer,
    'twitter2015': SATrainer2,
    'twitter2017': SATrainer2,
}
DATA_PROCESS = {
    'twitter15': (TVSAProcessor, TVSADataset),
    'twitter17': (TVSAProcessor, TVSADataset),
    'twitter2015': (TVSAProcessor2, TVSADataset2),
    'twitter2017': (TVSAProcessor2, TVSADataset2),
}

DATA_PATH = {
    'twitter15': {
        'train': 'data/twitter2015/train.txt',
        'dev': 'data/twitter2015/valid.txt',
        'test': 'data/twitter2015/test.txt',
        'train_auximgs': 'data/twitter2015/twitter2015_train_dict.pth',
        'dev_auximgs': 'data/twitter2015/twitter2015_val_dict.pth',
        'test_auximgs': 'data/twitter2015/twitter2015_test_dict.pth',
        # gcn data
        'gcn_train': 'data/twitter2015/twitter15_train.json',
        'gcn_dev': 'data/twitter2015/twitter15_dev.json',
        'gcn_test': 'data/twitter2015/twitter15_test.json'
    },
    'twitter17': {
        # text data
        'train': 'data/twitter2017/train.txt',
        'dev': 'data/twitter2017/valid.txt',
        'test': 'data/twitter2017/test.txt',
        'train_auximgs': 'data/twitter2017/twitter2017_train_dict.pth',
        'dev_auximgs': 'data/twitter2017/twitter2017_val_dict.pth',
        'test_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
        # gcn data
        'gcn_train': 'data/twitter2017/twitter17_train.json',
        'gcn_dev': 'data/twitter2017/twitter17_dev.json',
        'gcn_test': 'data/twitter2017/twitter17_test.json'
    },
    'twitter2015': {
        'train': 'data/twitter2015/twitter2015/train.txt',
        'dev': 'data/twitter2015/twitter2015/valid.txt',
        'test': 'data/twitter2015/twitter2015/test.txt',
        'train_auximgs': 'data/twitter2015/twitter2015_train_dict.pth',
        'dev_auximgs': 'data/twitter2015/twitter2015_val_dict.pth',
        'test_auximgs': 'data/twitter2015/twitter2015_test_dict.pth',
        'gcn_train': 'data/twitter2015/twitter15_train.json',
        'gcn_dev': 'data/twitter2015/twitter15_dev.json',
        'gcn_test': 'data/twitter2015/twitter15_test.json',
        # vao
        'path_img': 'data/twitter2015_images',
        'image_filename': 'data/ANP_data/image_output2015.json',
    },
    'twitter2017': {
        'train': 'data/twitter2017/twitter2017/train.txt',
        'dev': 'data/twitter2017/twitter2017/valid.txt',
        'test': 'data/twitter2017/twitter2017/test.txt',
        'train_auximgs': 'data/twitter2017/twitter2017_train_dict.pth',
        'dev_auximgs': 'data/twitter2017/twitter2017_val_dict.pth',
        'test_auximgs': 'data/twitter2017/twitter2017_test_dict.pth',
        'gcn_train': 'data/twitter2017/twitter17_train.json',
        'gcn_dev': 'data/twitter2017/twitter17_dev.json',
        'gcn_test': 'data/twitter2017/twitter17_test.json',
        # vao
        'image_filename': 'data/ANP_data/image_output2017.json',
        'path_img': 'data/twitter2017_images',
    },

}

IMG_PATH = {
    'twitter15': 'data/twitter2015_images',
    'twitter17': 'data/twitter2017_images',
    'twitter2015': 'data/twitter2015_images',
    'twitter2017': 'data/twitter2017_images',
}

AUX_PATH = {
    'twitter15': {
        'train': 'data/twitter2015_aux_images/train/crops',
        'dev': 'data/twitter2015_aux_images/val/crops',
        'test': 'data/twitter2015_aux_images/test/crops',
    },

    'twitter17': {
        'train': 'data/twitter2017_aux_images/train/crops',
        'dev': 'data/twitter2017_aux_images/val/crops',
        'test': 'data/twitter2017_aux_images/test/crops',
    },
    'twitter2015': {
        'train': 'data/twitter2015_aux_images/train/crops',
        'dev': 'data/twitter2015_aux_images/val/crops',
        'test': 'data/twitter2015_aux_images/test/crops',
    },

    'twitter2017': {
        'train': 'data/twitter2017_aux_images/train/crops',
        'dev': 'data/twitter2017_aux_images/val/crops',
        'test': 'data/twitter2017_aux_images/test/crops',
    },
}

AGN_PATH = {
    'twitter15': {
        'sum': 'data/AGN_data/twitter2015/aux.txt',
        'train': 'data/AGN_data/twitter2015/aux_train.txt',
        'dev': 'data/AGN_data/twitter2015/aux_dev.txt',
        'test': 'data/AGN_data/twitter2015/aux_test.txt',
    },
    'twitter17': {
        'sum': 'data/AGN_data/twitter2017/aux.txt',
        'train': 'data/AGN_data/twitter2017/aux_train.txt',
        'dev': 'data/AGN_data/twitter2017/aux_dev.txt',
        'test': 'data/AGN_data/twitter2017/aux_test.txt',
    },
    'twitter2015': {
        'sum': 'data/AGN_data/twitter2015/aux.txt',
        'train': 'data/AGN_data/twitter2015/aux_train.txt',
        'dev': 'data/AGN_data/twitter2015/aux_dev.txt',
        'test': 'data/AGN_data/twitter2015/aux_test.txt',
    },
    'twitter2017': {
        'sum': 'data/AGN_data/twitter2017/aux.txt',
        'train': 'data/AGN_data/twitter2017/aux_train.txt',
        'dev': 'data/AGN_data/twitter2017/aux_dev.txt',
        'test': 'data/AGN_data/twitter2017/aux_test.txt',
    },
}

MERGE_PATH = {
    'twitter2015': {
        'train': 'data/AGN_data/twitter2015/merge_train.txt',
        'dev': 'data/AGN_data/twitter2015/merge_dev.txt',
        'test': 'data/AGN_data/twitter2015/merge_test.txt',
    },
    'twitter2017': {
        'train': 'data/AGN_data/twitter2017/merge_train.txt',
        'dev': 'data/AGN_data/twitter2017/merge_dev.txt',
        'test': 'data/AGN_data/twitter2017/merge_test.txt',
    },
}

CAPTION_PATH = {
    'twitter2015': {
        'train': 'data/AGN_data/twitter2015/15_train_caption.txt',
        'dev': 'data/AGN_data/twitter2015/15_dev_caption.txt',
        'test': 'data/AGN_data/twitter2015/15_test_caption.txt',
    },
    'twitter2017': {
        'train': 'data/AGN_data/twitter2017/17_train_caption.txt',
        'dev': 'data/AGN_data/twitter2017/17_dev_caption.txt',
        'test': 'data/AGN_data/twitter2017/17_test_caption.txt',
    },
}

def set_seed(seed=2024):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='bert-base-uncased', type=str, help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=35, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--train_batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--eval_batch_size', default=16, type=int, help="eval batch size")
    parser.add_argument('--lr', default=5e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=16, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=1, type=int, help="random seed, default is 1")
    parser.add_argument('--prefix_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prefix_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default=None, type=str, help="save model at save_path")
    parser.add_argument('--use_pretrained', action='store_true', help="use pretrained NER model")
    parser.add_argument('--write_path', default=None, type=str,
                        help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prefix', action='store_true')
    parser.add_argument('--use_align', action='store_true')
    parser.add_argument('--only_caption', action='store_true')
    parser.add_argument('--use_probe', action='store_true')
    parser.add_argument('--use_152', action='store_true')
    parser.add_argument('--vao', action='store_true')
    parser.add_argument('--noauxloss', action='store_true')
    parser.add_argument('--gcn_layer_number', default=0, type=int, help="layers of gcn")
    parser.add_argument('--num_layers', type=int, default=0, help='Num of GCN layers.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='GCN layer dropout rate.')
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument('--do_aug', action='store_true', help="augument or not")
    parser.add_argument('--aug_type', default=None, type=str, help="span_cutoff, token_cutoff, dim_cutoff")
    parser.add_argument('--aug_cutoff_ratio', default=0.1, type=float, help="cutoff ratio")
    parser.add_argument('--aug_ce_loss', default=1.0, type=float, help="ce ratio")
    parser.add_argument('--aug_js_loss', default=1.0, type=float, help="js ratio")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--max_seq_agn', default=500, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")
    parser.add_argument("--cache_dir", default="data/image_cache_dir/")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased "
                             "models and False for cased models.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=12, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--logit_threshold", default=8., type=float,
                        help="Logit threshold for annotating labels.")
    parser.add_argument("--filter_type", default="f1", type=str, help="Which filter type to use")
    parser.add_argument("--use_heuristics", default=True, action='store_true',
                        help="If true, use heuristic regularization on span length")
    parser.add_argument("--use_nms", default=True, action='store_true',
                        help="If true, use nms to prune redundant spans")
    parser.add_argument("--verbose_logging", default=False, action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('-g', '--gpus', type=int, default=1, help="number of gpus per node.")
    parser.add_argument('--n_gpu', type=int, default=1, help="number of gpus.(sum)")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help="number of nodes.")

    args = parser.parse_args()
    data_path = DATA_PATH[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]
    aux_path = AUX_PATH[args.dataset_name]
    # agn_path = AGN_PATH[args.dataset_name]
    merge_path = MERGE_PATH[args.dataset_name]
    if args.only_caption:
        merge_path = CAPTION_PATH[args.dataset_name]

    set_seed(args.seed)
    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)
    print(args)

    logdir = "logs/" + args.dataset_name + "_" + str(args.train_batch_size) + "_" + str(args.lr) + args.notes
    writer = SummaryWriter(logdir=logdir)

    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if not args.use_prefix:
        aux_path = None
    if not args.use_align:
        processor = data_process(data_path, args.bert_name)
        train_dataset = dataset_class(processor, data_path['path_img'], aux_path, args.max_seq, args=args,
                                      sample_ratio=args.sample_ratio, mode='train')
        dev_dataset = dataset_class(processor, data_path['path_img'], aux_path, args.max_seq, args=args,
                                    sample_ratio=args.sample_ratio, mode='dev')
        test_dataset = dataset_class(processor, data_path['path_img'], aux_path, args.max_seq, args=args,
                                     sample_ratio=args.sample_ratio, mode='test')

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)

        train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler, num_workers=4,
                                      pin_memory=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True)

    else:
        processor = data_process(data_path, args.bert_name, merge_path)
        agn_train_dataset = dataset_class(processor, data_path['path_img'], aux_path, args.max_seq_agn, args=args,
                                          sample_ratio=args.sample_ratio, mode='train')
        agn_dev_dataset = dataset_class(processor, data_path['path_img'], aux_path, args.max_seq_agn, args=args,
                                        sample_ratio=args.sample_ratio, mode='dev')
        agn_test_dataset = dataset_class(processor, data_path['path_img'], aux_path, args.max_seq_agn, args=args,
                                         sample_ratio=args.sample_ratio, mode='test')

        print("agn_len", len(agn_dev_dataset), len(agn_dev_dataset[0]), len(agn_train_dataset[0][0]))

        if args.local_rank == -1:
            train_sampler = RandomSampler(agn_train_dataset)
        else:
            train_sampler = DistributedSampler(agn_train_dataset)
        train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
        agn_train_dataloader = DataLoader(agn_train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                                          num_workers=4, pin_memory=True)
        agn_dev_dataloader = DataLoader(agn_dev_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                        num_workers=4, pin_memory=True)
        agn_test_dataloader = DataLoader(agn_test_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)

    if args.dataset_name == 'twitter15' or args.dataset_name == 'twitter17':
        label_mapping = processor.get_label_mapping()
        label_list = list(label_mapping.keys())
    else:
        label_list = processor.get_labels()
        label_mapping = {label: idx for idx, label in enumerate(label_list, 1)}
    tokenizer = processor.tokenizer
    type_num = E2EASAOTProcessor().get_type_num()
    model = model_class(label_list, tokenizer, args, type_num)  # SAModel or SAModel2
    if args.use_align:
        trainer = Trainer(train_data=agn_train_dataloader, dev_data=agn_dev_dataloader,
                          test_data=agn_test_dataloader, model=model, label_map=label_mapping, args=args,
                          logger=logger, writer=writer, train_dataset=agn_train_dataset,
                          dev_dataset=agn_dev_dataset, test_dataset=agn_test_dataset, processor=processor)
    else:
        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader,
                          model=model,
                          label_map=label_mapping, args=args, logger=logger, writer=writer,
                          train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset,
                          processor=processor)

    if args.do_train:
        trainer.train()
        print("training and testing")
        # test best model
        f1 = trainer.test(args.num_epochs)

    if args.only_test:
        print("only testing")
        # only do test
        f1 = trainer.test(args.num_epochs)

    torch.cuda.empty_cache()
    writer.close()

if __name__ == "__main__":
    main()