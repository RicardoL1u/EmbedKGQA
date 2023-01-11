import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from collections import defaultdict
from transformers import RobertaTokenizer
import random
random.seed(42)

class DatasetAnonyQA(Dataset):
    def __init__(self, data, entity2idx,tokenizer):
        self.data = data
        self.entity2idx = entity2idx
        self.idx2entity = {v:k for k,v in  entity2idx.items()}
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.tokenizer = tokenizer
        self.mode = 'train'
        # self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # special_tokens = ['<spt>']
        # print(f'add speciall tokens {special_tokens}')
        # self.tokenizer.add_tokens(special_tokens,special_tokens=True)

    def __len__(self):
        return len(self.data)
    
    # def pad_sequence(self, arr, max_len=128):
    #     num_to_add = max_len - len(arr)
    #     for _ in range(num_to_add):
    #         arr.append('<pad>')
    #     return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point['text'] + ' <spt> ' + data_point['question']
        # no need to care about the over length seq since we have already done this in
        # https://github.com/RicardoL1u/rag/blob/master/prepro_openqa_dataset.py
        tokenized_result = self.tokenizer(
            question_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # head_id = self.entity2idx[data_point['topic_entity']]
        head_id = None
        for entity in data_point['topic_entity']:
            if entity in self.entity2idx.keys():
                head_id = self.entity2idx[entity]
                break

        if head_id is None:
            head_id = self.entity2idx[random.choice(list(self.entity2idx.keys()))]
        tail_ids = []
        not_in_kg = []
        for tail_name in data_point['ans_ids']:
            tail_name = tail_name.strip()
            #TODO: dunno if this is right way of doing things
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
            else:
                not_in_kg.append(tail_name)
        tail_onehot = self.toOneHot(tail_ids)
        if self.mode == 'train':
            return tokenized_result['input_ids'].squeeze(), tokenized_result['attention_mask'].squeeze(), torch.tensor(head_id), tail_onehot
        else:
            return tokenized_result['input_ids'].squeeze(), tokenized_result['attention_mask'].squeeze(), torch.tensor(head_id), tail_onehot, not_in_kg 

class DatasetMetaQA(Dataset):
    def __init__(self, data, entity2idx,tokenizer):
        self.data = data
        # self.entities = entities
        self.entity2idx = entity2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        # self.index_array = list(self.entities.keys())
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            #TODO: dunno if this is right way of doing things
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)
        return question_tokenized, attention_mask, head_id, tail_onehot 

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

