# %%
import pandas as pd
import logging 
import torch
import random
import numpy as np
import argparse

from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

# %%
class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None, alt_label=None) -> None:
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.alt_label = alt_label

# %%
class InputFeature(object):
    def __init__(
        self, 
        input_ids, 
        attn_mask, 
        src_mask, 
        trg_mask, 
        label_id, 
        alt_label_id, 
        segment_ids=None,
        guid=None 
    ):

        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.src_mask = src_mask
        self.trg_mask = trg_mask
        self.label_id = label_id
        self.alt_label_id = alt_label_id 
        self.segment_id = segment_ids
        self.guid = guid 

# %%
class FullDataProcessor():
    @classmethod
    def _read_csv_pd(cls, input_file):
        reader = pd.read_csv(input_file, sep=',', encoding='utf8')
        return reader
    
    def _create_examples(self, datasets, data_type):
        train_data, dev_data, test_data = [], [], []
        guid = 0
        for line in tqdm(datasets.iterrows(), total=len(datasets), desc='loading examples...'):
            relation = line[1].loc['Relation']
            if relation not in ['Explicit', 'Implicit']:
                continue
            if data_type == 'Trans' and relation != 'Implicit':
                continue
            section = line[1].loc['Section']
            text_a = line[1].loc['Arg1_RawText']
            text_b = line[1].loc['Arg2_RawText']
            label = line[1].loc['ConnHeadSemClass1'].split('.')[0]
            try:
                alt_label = line[1].loc['ConnHeadSemClass2'].split('.')[0]
            except:
                alt_label = label
            
            if int(section) < 2:
                dev_data.append(
                    InputExample(
                        guid=guid, 
                        text_a=text_a, 
                        text_b=text_b, 
                        label=label, 
                        alt_label=alt_label, 
                        )
                    )
            elif int(section) > 20 and int(section) < 23:
                test_data.append(
                    InputExample(
                        guid=guid, 
                        text_a=text_a, 
                        text_b=text_b, 
                        label=label, 
                        alt_label =alt_label,
                        )
                    )
            else:
                train_data.append(
                    InputExample(
                        guid=guid, 
                        text_a=text_a, 
                        text_b=text_b, 
                        label=label, 
                        alt_label=alt_label,
                        )
                    )
            guid += 1

        return train_data, dev_data, test_data

    def get_examples(self, input_file, data_type='Full'):
        return self._create_examples(
            self._read_csv_pd(
                input_file
            ), 
            data_type
        )

# %%
def attention_masks(input_ids):
    atten_masks =  [float(i != 1) for i in input_ids]
    return np.array(atten_masks, dtype=int)


def make_src_mask(src, src_pad_idx=1):
    src = np.concatenate((src[:64], src[128:192]), axis=0)
    src_mask = (src != src_pad_idx).reshape(1, 1, len(src))
    # (1, 1, src_len)
    return np.array(src_mask, dtype=int)

def make_trg_mask(trg):
    trg_len = len(trg) // 2
    # trg = torch.Tensor(trg)
    # trg_len = trg.shape
    trg_mask = np.tril(np.ones((trg_len, trg_len))).reshape(1, trg_len, trg_len)
    return np.array(trg_mask, dtype=int)

# %%
def _truncate_seq_pair_dz(tokens_a, tokens_b, max_single_length):
    len_a = max_single_length
    len_b = max_single_length
    if len(tokens_a) < max_single_length:
        len_a = len(tokens_a)
        tokens_a = tokens_a + ['<pad>'] * (max_single_length - len_a)
    else:
        tokens_a = tokens_a[:max_single_length]
    
    if len(tokens_b) < max_single_length:
        len_b = len(tokens_b)
        tokens_b = tokens_b + ['<pad>'] * (max_single_length - len_b)
    else:
        tokens_b = tokens_b[:max_single_length]

    return tokens_a, tokens_b, len_a, len_b

# %%
def convert_text_to_token(tokenizer, arg1, arg2, max_single_length):
    tokens_a = tokenizer.tokenize(arg1)  # 分词
    tokens_b = tokenizer.tokenize(arg2) 
    tokens_a, tokens_b, len_a, len_b = _truncate_seq_pair_dz(tokens_a, tokens_b, max_single_length)
    tokens = ['<s>'] + tokens_a + ['</s>'] + ['</s>'] + tokens_b + ['</s>']
        
    tokens = tokenizer.convert_tokens_to_ids(tokens)

    return np.array(tokens)

# %%
def convert_examples_to_roberta_features(tokenizer, examples, task, max_seq_len, dataset_type='trian'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples_num_map = {
        'Temporal': 704, 
        'Contingency': 2104, 
        'Comparison': 3622, 
        'Expansion':7394
    }

    label_list=[task]
    label_map = {
            label_list[0]: 1
        }   # 不是该标签的都为0（二分类）
    if task == 'pdtb2_4':
        label_list=["Expansion", "Contingency", "Comparison", "Temporal"]
        label_map = {
                label_list[0]: 0,
                label_list[1]: 1,
                label_list[2]: 2, 
                label_list[3]: 3
            }  

    elen = len(examples)
    features = []

    for (ex_idx, example) in tqdm(enumerate(examples), total=len(examples), desc='building features...'):
        #训练集下采样每个类别采样率为1：1：1：1
        if task != 'pdtb2_4' and min(examples_num_map[task] / examples_num_map[example.label], 1) < random.random() and dataset_type=='train':
            continue 
        
        input_ids = convert_text_to_token(tokenizer, example.text_a, example.text_b, max_seq_len)   

        attn_mask = attention_masks(input_ids)
        src_mask = make_src_mask(input_ids)
        trg_mask = make_trg_mask(input_ids)
        segment_ids = []    # RoBERTa用不到
        
        label_id = label_map.get(example.label, 0)
        alt_label_id = label_map.get(example.label, 0)
        
    
        features.append(InputFeature(input_ids=input_ids,
                                    attn_mask=attn_mask,
                                    src_mask=src_mask, 
                                    trg_mask=trg_mask, 
                                    segment_ids=segment_ids,
                                    label_id=label_id,
                                    alt_label_id=alt_label_id,
                                    guid=example.guid))

    return features

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', default='./data/', type=str, required=False)
    parser.add_argument('--max_seq_len', default=126, type=int, required=False)
    args = parser.parse_args()

    full_train_data, _, _ = FullDataProcessor().get_examples(args.data_dir + 'pdtb2.csv')
    
    features = convert_examples_to_roberta_features(tokenizer, full_train_data, 'pdtb2_4', args.max_seq_len)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attn_mask = torch.tensor([f.attn_mask for f in features], dtype=torch.long)
    all_src_mask = torch.tensor([f.src_mask for f in features], dtype=torch.long)
    all_trg_mask = torch.tensor([f.trg_mask for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_alt_label_id = torch.tensor([f.alt_label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attn_mask, all_src_mask, all_trg_mask, all_label_id, all_alt_label_id)
    print("Full dataset lens:{}".format(len(dataset)))
    torch.save(dataset, args.data_dir + 'Full_' + 'pdtb2_4' + '_train')
    
    train_data, dev_data, test_data = FullDataProcessor().get_examples(args.data_dir + 'pdtb2.csv', data_type='Trans')
    for task in ['pdtb2_4', 'Temporal', 'Comparison', 'Contingency', 'Expansion']:
        features = convert_examples_to_roberta_features(tokenizer, train_data, task, args.max_seq_len)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attn_mask = torch.tensor([f.attn_mask for f in features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_alt_label_id = torch.tensor([f.alt_label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attn_mask, all_label_id, all_alt_label_id)
        print("{} dataset lens:{}".format(task, len(dataset)))
        torch.save(dataset, args.data_dir + task + '/Trans_train')

    for task in ['pdtb2_4', 'Temporal', 'Comparison', 'Contingency', 'Expansion']:
        features = convert_examples_to_roberta_features(tokenizer, dev_data, task, args.max_seq_len, dataset_type='dev')
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attn_mask = torch.tensor([f.attn_mask for f in features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_alt_label_id = torch.tensor([f.alt_label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attn_mask, all_label_id, all_alt_label_id)
        torch.save(dataset, args.data_dir + task + '/Trans_dev')
    
    print("dev dataset lens:{}".format(len(dataset)))

    for task in ['pdtb2_4', 'Temporal', 'Comparison', 'Contingency', 'Expansion']:
        features = convert_examples_to_roberta_features(tokenizer, test_data, task, args.max_seq_len, dataset_type='test')
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attn_mask = torch.tensor([f.attn_mask for f in features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_alt_label_id = torch.tensor([f.alt_label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attn_mask, all_label_id, all_alt_label_id)
        torch.save(dataset, args.data_dir + task + '/Trans_test')
    
    print("test dataset lens:{}".format(len(dataset)))


# %%



