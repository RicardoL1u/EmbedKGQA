import json
import os
from tqdm import tqdm
import requests
import re
import pickle
def get_predicted_url(title:str)->str:
    prefix = 'https://en.wikipedia.org/wiki/'
    title = title.replace(' ','_')
    title = title.replace('?', '%3F')
    # title = title.replace('&amp;', '%26')
    # title = title.replace('\'','%27')
    return prefix+title

data_path = '/data/lyt/exp/acl'
topic_entity_set = set()

with open("KGQA/RoBERTa/complex_wikidata5m.pkl", "rb") as fin:
    wiki5m = pickle.load(fin)
entity2idx = wiki5m.graph.entity2id
entity_5m_set = set(entity2idx.keys())
# idx2entity = {v:k for k,v in  entity2idx.items()}
# embedding_matrix = torch.tensor(wiki5m.solver.entity_embeddings)


url2qid_map = {}
# eni2qid_url_2018 = json.load(open('/data/lyt/wikidata-full/eni2qid_url_2018.json'))

# for _,v in eni2qid_url_2018.items():
#     assert type(v) == list and len(v) == 2
#     url2qid_map[v[1]] = v[0]

with open('url2qid.json') as f:
    url2qid_map = json.load(f)

topic_entities = {
    'train':[],
    'valid':[],
    'iid_test':[],
    'ood_test':[],
}

regex = "EntityPage\/(Q[0-9]+)"


for phase in ['iid_test','ood_test','valid','train']:
    datas = json.load(open(os.path.join(data_path,f'link_{phase}.json')))
    topic_entities[phase] = [None] * len(datas)
    for idx,data in tqdm(zip(range(len(datas)),datas)):
        mark = False
        for tuple_str in data['pred_tuples_string']:
            entity_name = tuple_str[0]
            entity_url = get_predicted_url(entity_name)
            if entity_url in url2qid_map and url2qid_map[entity_url] in entity_5m_set:
                qid = url2qid_map[entity_url]
                topic_entity_set.add(qid)
                if topic_entities[phase][idx] == None:
                    topic_entities[phase][idx] = [qid]
                else:
                    topic_entities[phase][idx].append(qid)
                mark = True
                # break
        if not mark:
            print('Failed once')
            for tuple_str in data['pred_tuples_string']:
                entity_name = tuple_str[0]
                entity_url = get_predicted_url(entity_name)
                # check
                if entity_url in url2qid_map.keys():
                    continue
                try:
                    r = requests.get(entity_url)
                except:
                    # failed_link.append(url)
                    continue
                qids = re.findall(regex,r.text)
                if type(qids) == list and len(qids) > 0:  
                    url2qid_map[entity_url]=qids[0]
                    print('insert one!')
                    if topic_entities[phase][idx] == None and qids[0] in entity_5m_set:
                        topic_entities[phase][idx] = [qids[0]]
                        topic_entity_set.add(qids[0])
                        print('find one!')
                else:
                    print(qids)                

with open('url2qid.json','w') as f:
    json.dump(url2qid_map,f,indent=4,ensure_ascii=False)

assert all(topic_entities['train']) and all(topic_entities['valid']) and all(topic_entities['iid_test']) and all(topic_entities['ood_test'])
assert topic_entity_set.issubset(entity_5m_set)

with open(os.path.join(data_path,'link_result.json'),'w') as f:
    json.dump(topic_entities,f,indent=4,ensure_ascii=False)

# with open('topic_entity.json','w') as f:
#     json.dump(list(topic_entity_set),f,indent=4,ensure_ascii=False)