import argparse
import re
import numpy as np
import json
import pickle
from tqdm import tqdm
with open("complex_wikidata5m.pkl", "rb") as fin:
    model = pickle.load(fin)
entity2id = model.graph.entity2id
relation2id = model.graph.relation2id
entity_embeddings = model.solver.entity_embeddings
relation_embeddings = model.solver.relation_embeddings

data_path = '/data/wangyifan/TransferNet/data/AnonyQA/'

file_list = ['2-hop-triplet-x.txt','2-hop-triplet-y.txt','3-hop-triplet-x.txt','3-hop-triplet-y.txt']

entity_set = set()
for file_name in file_list:
    with open(data_path+file_name) as f:
        lines = f.readlines()
    for line in tqdm(lines):
        e1,e2 = re.findall(r'Q[0-9]+',line)
        assert type(e1) == str and type(e2) == str and len(e1)>1 and len(e2)>1
        entity_set.add(e1)
        entity_set.add(e2)
print(f'the entity num of big5m by wyf is {len(entity_set)}')

with open('/data/lyt/docRed/refine/ans_list.json') as f:
    ans_set = set(json.load(f))

topic_entity_set = set(json.load(open('/data/lyt/exp/EmbedKGQA/topic_entity.json')))

correct_part = entity_set.intersection(ans_set)
print(f'intersection len: {len(correct_part)}')
print(f'sub KG len: {len(entity_set)}')
print(f'ans len: {len(ans_set)}')
print(f'acc: {len(correct_part)/len(entity_set)}')
print(f'recall: {len(correct_part)/len(ans_set)}')

all_set = entity_set.union(topic_entity_set)

entity_dict = {}
for entity in all_set:
    idx = entity2id[entity]
    entity_dict[entity] = entity_embeddings[idx]


complex_wyf_big5m = argparse.Namespace()
complex_wyf_big5m.graph = argparse.Namespace()
complex_wyf_big5m.solver = argparse.Namespace()
complex_wyf_big5m.graph.entity2id = {}
complex_wyf_big5m.solver.entity_embeddings = []
idx = 0 
for k,v in entity_dict.items():
    complex_wyf_big5m.graph.entity2id[k] = idx
    complex_wyf_big5m.solver.entity_embeddings.append(v)
    idx += 1

complex_wyf_big5m.solver.entity_embeddings = np.array(complex_wyf_big5m.solver.entity_embeddings)


with open('complex_wyf_big5m.pkl','wb') as f:
    pickle.dump(complex_wyf_big5m,f)


