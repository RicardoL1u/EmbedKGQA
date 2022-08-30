from distutils.errors import LinkError
import json
import os
phases = ['train','eval','test']
data_path = '/data/lyt/exp/EmbedKGQA'

print('loading wikidata qid2entity_name')
with open('/data/lyt/wikidata-full/wikidata-item-en-label.json') as f:
    qid2entity_name_map = json.load(f)
print('loaded qid 2 entity name')

link_result = json.load(open(data_path+'/link_result.json'))
pre_q_ans = json.load(open(data_path+'/per_q_ans.json'))
failed_entities = []
for phase in phases:
    datas = json.load(open(data_path+f'/{phase}.json'))
    assert len(datas) == len(pre_q_ans[phase]), print(phase,len(datas),len(pre_q_ans[phase])) 
    assert len(datas) == len(link_result[phase]), print(phase,len(datas),len(link_result[phase]))
    for data,topic_entity,q_ans in zip(datas,link_result[phase],pre_q_ans[phase]):
        data['topic_entity'] = topic_entity
        data['ans_ids'] = q_ans
        if topic_entity not in qid2entity_name_map.keys():
            failed_entities.append(topic_entity)
            continue
        data['topic_entity_name'] = qid2entity_name_map[topic_entity]
    
    with open(data_path+f'/{phase}.json','w') as f:
        json.dump(datas,f,indent=4,ensure_ascii=False)

print(f'fail {len(failed_entities)} entities')
with open('/data/lyt/wikidata-full/failed_in_wikidata_en_label.json','w') as f:
    json.dump(failed_entities,f,indent=4)
