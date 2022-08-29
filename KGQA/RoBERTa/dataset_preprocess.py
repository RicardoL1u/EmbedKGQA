from distutils.errors import LinkError
import json
import os
phases = ['train','eval','test']
data_path = '/data/lyt/exp/EmbedKGQA'

link_result = json.load(open(data_path+'/link_result.json'))
pre_q_ans = json.load(open(data_path+'/per_q_ans.json'))

for phase in phases:
    datas = json.load(open(data_path+f'/{phase}.json'))
    assert len(datas) == len(pre_q_ans[phase]), print(phase,len(datas),len(pre_q_ans[phase])) 
    assert len(datas) == len(link_result[phase]), print(phase,len(datas),len(link_result[phase]))
    for data,topic_entity,q_ans in zip(datas,link_result[phase],pre_q_ans[phase]):
        data['topic_entity'] = topic_entity
        data['ans_ids'] = q_ans
    
    with open(data_path+f'/{phase}.json','w') as f:
        json.dump(datas,f,indent=4,ensure_ascii=False)
    
