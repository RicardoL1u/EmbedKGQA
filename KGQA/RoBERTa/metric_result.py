import json
from single import Metric
datapath = '/data/lyt/exp/rush/embedKGQA/human/'
import pickle
kg_name = 'complex_wyf_big5m'
wiki5m = pickle.load(open(kg_name+'.pkl','rb'))
entity2idx = wiki5m.graph.entity2id

print('loading wikidata qid2entity_name')
with open('/data/lyt/wikidata-5m/wikidata-5m-entity-en-label.json') as f:
    qid2entity_name_map = json.load(f)
print('loaded qid 2 entity name')

for file in [f'{kg_name}_small_iid_test.json',f'{kg_name}_small_ood_test.json',f'{kg_name}_valid.json']:
    dataset = json.load(open(datapath+file))
    total_pred_list = []
    for data in dataset:
        if 'pred_names' not in data.keys():
            data['pred_names'] = [qid2entity_name_map[qid] for qid in data['pred_qids']]
        total_pred_list.append(list(set([pred['label'] for pred in data['pred_names']])))
    assert len(data['answers']) == len(set(data['ans_ids']))
    total_ans_list = [list(set(data['answers'])) for data in dataset]
    if 'pred_names' not in data.keys():
        with open(datapath+file,'w') as f:
            json.dump(dataset,f,indent=4,ensure_ascii=False)
    print(Metric.str_metric(total_pred_list,total_ans_list))

# for file in ['train','valid','iid_test','ood_test']:
#     dataset = json.load(open('/data/lyt/exp/acl/embedKGQA/'+file+'.json'))
#     for data in dataset:
#         for ans in data['ans_ids']:
#             if ans not in entity2idx.keys():
#                 print('hi')
