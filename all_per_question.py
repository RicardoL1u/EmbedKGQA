import json
import os
ans_list = {
    'train':[],
    'test':[],
    'eval':[]
}
data_path = '/data/lyt/docRed/refine'
for phase in ['eval','test','train']:
    doc_list = json.load(open(os.path.join(data_path,f'{phase}.json')))
    for doc in doc_list:
        for q in doc['questions']:    
            q_ans_list = []
            for ans in q['ans']:
                assert type(ans) == list and len(ans) == 2
                q_ans_list.append(ans[0])
            ans_list[phase].append(q_ans_list)

with open(os.path.join('/data/lyt/exp/rag','per_q_ans.json'),'w') as f:
    json.dump(ans_list,f,indent=4,ensure_ascii=False)