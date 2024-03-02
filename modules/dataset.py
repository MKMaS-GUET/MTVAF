import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
from transformers import BertTokenizer,RobertaTokenizer
from torchvision import transforms
import logging

from models.utils import read_absa_data, convert_absa_data, convert_examples_to_features, read_agn_data, image_process

logger = logging.getLogger(__name__)

class TVSAProcessor(object):
    def __init__(self, data_path, bert_name, merge_path=None) -> None:
        self.data_path = data_path
        self.merge_path = merge_path
        if "roberta" in bert_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(bert_name, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)


    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # load aux image
        aux_path = self.data_path[mode + "_auximgs"]
        aux_imgs = torch.load(aux_path)

        dataset = read_absa_data(load_file, sample_ratio)
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(dataset["words"]))), k=int(len(dataset["words"]) * sample_ratio))
            sample_sentences = [dataset["sentences"][idx] for idx in sample_indexes]
            sample_words = [dataset["words"][idx] for idx in sample_indexes]
            sample_ote_targets = [dataset["ote_targets"][idx] for idx in sample_indexes]
            sample_ts_targets = [dataset["ts_targets"][idx] for idx in sample_indexes]
            sample_imgs = [dataset["imgs"][idx] for idx in sample_indexes]
            assert len(sample_sentences) == len(sample_words) == len(sample_ote_targets) == len(sample_ts_targets)  == len(sample_imgs), \
                "{}, {}, {}, {}, {}".format(
                    len(sample_sentences), len(sample_words), len(sample_ote_targets), len(sample_ts_targets), len(sample_imgs))
            return {"sentences": sample_sentences, "words": sample_words, "ote_targets": sample_ote_targets,
                    "ts_targets": sample_ts_targets, "imgs": sample_imgs, "aux_imgs": aux_imgs}

        dataset["aux_imgs"]=aux_imgs
        if self.merge_path:
            agn_file = self.merge_path[mode]
            logger.info("Loading agn data from {}".format(agn_file))
            agn_dataset = read_agn_data(agn_file, dataset, mode)
            agn_dataset["aux_imgs"] = aux_imgs
            return agn_dataset

        return dataset
    def get_label_mapping(self):
        # LABEL_LIST = ["O", "T", "T-POS", "T-NEG", "T-NEU", "T-NEG-B", "T-NEU-B", "T-POS-B", "X", "[CLS]", "[SEP]"]
        # LABEL_LIST = ['other', 'neutral', 'positive', 'negative', 'conflict']
        LABEL_LIST = ['O', 'EQ', 'B-POS', 'I-POS', 'E-POS', 'S-POS', 'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG','B-NEU', 'I-NEU', 'E-NEU', 'S-NEU', "[CLS]", "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST,0)}
        return label_mapping


class TVSADataset(Dataset):
    def __init__(self, processor, img_path=None, aux_img_path=None, max_seq=40, args=None, sample_ratio=1, mode='train',
                 ignore_idx=0) -> None:
        self.processor = processor
        self.data_dict = processor.load_from_file(mode, sample_ratio) #dataset dict
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else None
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.args = args

        self.examples = convert_absa_data(self.img_path, dataset=self.data_dict, args=self.args,verbose_logging=self.args.verbose_logging)  # transform  the data into the example class
        self.features = convert_examples_to_features(self.examples, self.tokenizer, self.max_seq,self.args.verbose_logging, logger)

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        self.all_start_positions = torch.tensor([f.start_positions for f in self.features], dtype=torch.long)
        self.all_end_positions = torch.tensor([f.end_positions for f in self.features], dtype=torch.long)
        self.all_bio_labels = torch.tensor([f.bio_labels for f in self.features], dtype=torch.long)
        self.all_polarity_positions = torch.tensor([f.polarity_positions for f in self.features], dtype=torch.long)
        self.all_example_index = torch.arange(self.all_input_ids.size(0), dtype=torch.long)  # 0-509

        logger.info("Num orig examples = %d", len(self.examples))
        logger.info("Num split features = %d", len(self.features))
        logger.info("Train Batch size = %d", self.args.train_batch_size)
        logger.info("Eval Batch size = %d", self.args.eval_batch_size)

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        img = self.data_dict['imgs'][idx][0]
        data = self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], self.all_example_index[idx], self.all_start_positions[idx], self.all_end_positions[idx], self.all_bio_labels[idx], self.all_polarity_positions[idx]
        if self.img_path is not None and self.args.use_prefix:

            try:
                img_path = os.path.join(self.img_path, img)
                image = image_process(img_path)
            except:
                img_path_ = self.img_path+"/"+img
                print(" Can not find image {}".format(img_path_))
                img_path = os.path.join(self.img_path, '17_06_4705.jpg')
                image = image_process(img_path)

            if self.aux_img_path is not None:
                aux_imgs = []
                aux_img_paths = []
                if img in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][img]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = image_process(aux_img_paths[i])
                    aux_imgs.append(aux_img)

                for i in range(3 - len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
                data = self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], self.all_example_index[idx], \
                       self.all_start_positions[idx], self.all_end_positions[idx], self.all_bio_labels[idx], \
                       self.all_polarity_positions[idx], image, aux_imgs
                return data
        return data


class TVSAProcessor2(object):
    def __init__(self, data_path, bert_name, merge_path=None) -> None:
        self.data_path = data_path
        self.merge_path = merge_path
        if "roberta" in bert_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(bert_name, do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)

    def _read_mmtsv(cls, filename, image_filename, path_img):
        with open(image_filename, 'r') as f:
            image_data = json.load(f)
        f = open(filename, encoding='utf-8')
        data = []
        sentence = []
        label= []

        imgs = []
        auxlabels = []
        auxlabel = []
        imagelabels = []
        imgid = ''
        count = 0
        for line in f:
            if line.startswith('IMGID:'):
                imgid = line.strip().split('IMGID:')[1]+'.jpg'
                continue
            if line[0]=="\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    imgs.append(imgid)
                    image_path = os.path.join(path_img, imgid)
                    if not os.path.exists(image_path):
                        print(image_path)
                    try:
                        image = image_process(image_path)
                    except:
                        imgid = '17_06_4705.jpg'

                    image_label = image_data.get(imgid)
                    if image_label == None:
                        count += 1
                    auxlabels.append(auxlabel)
                    imagelabels.append(image_label)
                    sentence = []
                    label = []
                    imgid = ''
                    auxlabel = []
                continue
            splits = line.split('\t')
            sentence.append(splits[0])
            cur_label = splits[1]
            if cur_label == 'B-OTHER':
                cur_label = 'B-MISC'
            elif cur_label == 'I-OTHER':
                cur_label = 'I-MISC'
            label.append(cur_label)
            auxlabel.append(cur_label)

        print("The number of samples with NULL image labels: "+ str(count))
        if len(sentence) >0:
            data.append((sentence, label))
            imgs.append(imgid)
            auxlabels.append(auxlabel)
            imagelabels.append(image_label)

        print("The number of samples: "+ str(len(data)))
        print("The number of images: "+ str(len(imgs)))
        return data, imgs, auxlabels, imagelabels

    def get_labels(self):
        return ["O", "B-NEU", "I-NEU", "B-POS", "I-POS", "B-NEG", "I-NEG","X","[CLS]","[SEP]"]

    ### modify
    def get_auxlabels(self):
        return ["O", "B-NEU", "I-NEU", "B-POS", "I-POS", "B-NEG", "I-NEG", "X", "[CLS]", "[SEP]"]

    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[CLS]']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['[SEP]']

    def _create_examples(self, lines, imgs, auxlabels, imagelabels, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            imagelabel = imagelabels[i]
            examples.append(MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel, imagelabel = imagelabel))
        return examples

    def _create_examples2(self, lines, imgs, auxlabels, imagelabels, set_type):
        examples = []
        visual_context = {}
        with open(self.merge_path[set_type], 'r', encoding='utf-8') as f:
            for line in f:
                img, merge_text = line.strip().split(" [SEP] ", 1)
                visual_context[img] = merge_text
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            try:
                text_a = text_a.strip() + " [SEP] " + visual_context[img_id].strip()
            except KeyError:
                print("visual_context do not have", img_id)
            label = label
            auxlabel = auxlabels[i]
            imagelabel = imagelabels[i]
            examples.append(MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel, imagelabel = imagelabel))
        return examples

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # load aux image
        aux_path = self.data_path[mode + "_auximgs"]
        aux_imgs = torch.load(aux_path)
        data, imgs, auxlabels, imagelabels = self._read_mmtsv(load_file, self.data_path['image_filename'], self.data_path['path_img'])

        examples = self._create_examples(data, imgs, auxlabels, imagelabels, mode)
        # print(dataset)
        if self.merge_path:
            examples = self._create_examples2(data, imgs, auxlabels, imagelabels, mode)
        return examples, aux_imgs


class TVSADataset2(Dataset):
    def __init__(self, processor, img_path=None, aux_img_path=None, max_seq=40, args=None, sample_ratio=1, mode='train',
                 ignore_idx=0) -> None:
        self.processor = processor

        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_labels()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else None
        self.mode = mode
        self.args = args

        self.examples, self.aux_imgss = processor.load_from_file(mode)

        if args.use_align:
            self.features = convert_merge_examples_to_features(self.examples, processor.get_labels(),  processor.get_auxlabels(), max_seq, processor.tokenizer, img_path, self.aux_img_path, self.aux_imgss)
        else:
            self.features = convert_mm_examples_to_features(self.examples, processor.get_labels(),processor.get_auxlabels(), max_seq, processor.tokenizer, img_path, self.aux_img_path, self.aux_imgss)
        self.all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        self.all_added_input_mask = torch.tensor([f.added_input_mask for f in self.features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        self.all_images = torch.stack([f.images for f in self.features])
        if self.aux_img_path is not None:
            self.all_aux_imgs = torch.stack([f.aux_imgs for f in self.features])
        else:
            self.all_aux_imgs = torch.zeros_like(self.all_segment_ids)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features], dtype=torch.long)
        self.all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in self.features], dtype=torch.long)
        self.all_imagelabel = torch.tensor([f.imagelabel for f in self.features], dtype=torch.float)

        logger.info("Num orig examples = %d", len(self.examples))
        logger.info("Num split features = %d", len(self.features))
        logger.info("Train Batch size = %d", self.args.train_batch_size)
        logger.info("Eval Batch size = %d", self.args.eval_batch_size)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        img = self.examples[idx].img_id
        if self.args.use_prefix:
            data = self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], self.all_label_ids[idx], self.all_auxlabel_ids[idx], self.all_imagelabel[idx], self.all_images[idx], self.all_aux_imgs[idx]
        else:
            data = self.all_input_ids[idx], self.all_input_mask[idx], self.all_segment_ids[idx], self.all_label_ids[idx]

        return data

class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None,imagelabel= None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        self.auxlabel = auxlabel
        self.imagelabel = imagelabel


class MMInputFeatures(object):

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, images, aux_imgs, label_id, auxlabel_id, imagelabel):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.images = images
        self.aux_imgs = aux_imgs
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id
        self.imagelabel = imagelabel

def convert_mm_examples_to_features(examples, label_list, auxlabel_list, max_seq_length, tokenizer, path_img, aux_img_path, aux_imgss):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        imagelabellist = example.imagelabel
        imagelabellist = dict(sorted(imagelabellist.items()))
        imagelabel_value =[0]* len(imagelabellist)
        for i, (k, v) in enumerate(imagelabellist.items()):
            imagelabel_value[i]= v
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            word = " "+ word
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length

        image_name = example.img_id
        image_path = path_img +'/'+ image_name

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path)
        except:
            print(" Can not find image {}".format(image_path))
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail)

        aux_imgs = []
        if aux_img_path is not None:
            aux_img_paths = []
            if image_name in aux_imgss:
                aux_img_paths = aux_imgss[image_name]
                aux_img_paths = [os.path.join(aux_img_path, path) for path in aux_img_paths]
            for i in range(min(3, len(aux_img_paths))):
                aux_img = image_process(aux_img_paths[i])
                aux_imgs.append(aux_img)

            for i in range(3 - len(aux_img_paths)):
                aux_imgs.append(torch.zeros((3, 224, 224)))

            aux_imgs = torch.stack(aux_imgs, dim=0)
            assert len(aux_imgs) == 3

        if ex_index < 2: #
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                          segment_ids=segment_ids, images=image, aux_imgs=aux_imgs, label_id=label_ids, auxlabel_id=auxlabel_ids, imagelabel= imagelabel_value))

    print('the number of problematic samples: ' + str(count))

    return features

def convert_merge_examples_to_features(examples, label_list, auxlabel_list, max_seq_length, tokenizer, path_img, aux_img_path, aux_imgss):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        text_1 = example.text_a.split(' [SEP] ', 1)[0].split(' ')
        text_len = len(text_1)

        labellist = example.label
        auxlabellist = example.auxlabel
        imagelabellist = example.imagelabel
        imagelabellist = dict(sorted(imagelabellist.items()))
        imagelabel_value =[0]* len(imagelabellist)
        for i, (k, v) in enumerate(imagelabellist.items()):
            imagelabel_value[i]= v
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            word = " "+ word
            token = tokenizer.tokenize(word)
            tokens.extend(token)

            label_1 = labellist[i] if i < text_len else labellist[0]
            auxlabel_1 = auxlabellist[i] if i < text_len else auxlabellist[0]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["[CLS]"])
        auxlabel_ids.append(auxlabel_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])
        auxlabel_ids.append(auxlabel_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length

        image_name = example.img_id
        image_path = path_img +'/'+ image_name

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path)
        except:
            print(" Can not find image {}".format(image_path))
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail)

        aux_imgs = []
        if aux_img_path is not None:
            aux_img_paths = []
            if image_name in aux_imgss:
                aux_img_paths = aux_imgss[image_name]
                aux_img_paths = [os.path.join(aux_img_path, path) for path in aux_img_paths]
            for i in range(min(3, len(aux_img_paths))):
                aux_img = image_process(aux_img_paths[i])
                aux_imgs.append(aux_img)

            for i in range(3 - len(aux_img_paths)):
                aux_imgs.append(torch.zeros((3, 224, 224)))

            aux_imgs = torch.stack(aux_imgs, dim=0)
            assert len(aux_imgs) == 3  # RGB #torch.Size([3, 3, 224, 224])

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                          segment_ids=segment_ids, images=image, aux_imgs=aux_imgs, label_id=label_ids, auxlabel_id=auxlabel_ids, imagelabel= imagelabel_value))

    print('the number of problematic samples: ' + str(count))

    return features