import json
import re
phases = ['train','valid','small_iid_test','small_ood_test']
data_path = '/data/lyt/exp/rush/embedKGQA'

# print('loading wikidata qid2entity_name')
# with open('/data/lyt/wikidata-full/wikidata-item-en-label.json') as f:
#     qid2entity_name_map = json.load(f)
# print('loaded qid 2 entity name')


# pre_q_ans = json.load(open(data_path+'/per_q_ans.json'))
# failed_entities = []

question_model_list = ['template']

for question_model in question_model_list:
    question_key = 'question'
    if question_model == 'gpt':
        question_key = 'gpt_best_question'
    elif question_model == 'template':
        question_key = 'template_question'

    for phase in phases:
        datas = json.load(open('/data/lyt/exp/rush/kgqa'+f'/{phase}.json'))
        new_datas = []
        for data in datas:
            names = re.findall(r'(\[.*?\])',data['context'])
            new_datas.append(
                {
                    'id':data['id'],
                    'topic_entity':data[f'linked_topic_entity_for_{question_model}'],
                    'answers':data['answers'],
                    'ans_ids':data['answer_ids'],
                    'text':data['context'],
                    'question':data[question_key],
                    'title':data['title'] if len(names) == 0 else names[0],
                }
            )

        with open(data_path+f'/{question_model}/{phase}.json','w') as f:
            json.dump(new_datas,f,indent=4,ensure_ascii=False)

# print(f'fail {len(failed_entities)} entities')
# with open('/data/lyt/wikidata-full/failed_in_wikidata_en_label.json','w') as f:
#     json.dump(failed_entities,f,indent=4)
