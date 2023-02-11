# TODO: THIS CODE IS ASSUME THE LINK OF HUMAN-ANNOTATED QUETSION HAS BEEN DONE

import json

gpt_id2topic_map = json.load(open('/data/lyt/exp/rush/kgqa/link_gpt_best_question_result.json'))
template_id2topic_map = json.load(open('/data/lyt/exp/rush/kgqa/link_template_question_result.json'))


for file in ['train','valid','small_iid_test','small_ood_test']:
    dataset = json.load(open('/data/lyt/exp/rush/embedKGQA/'+file+'.json'))
    for data in dataset:
        data['linked_topic_entity_for_human'] = data.pop('linked_topic_entity')
        data['linked_topic_entity_for_gpt'] = gpt_id2topic_map[data['id']]
        data['linked_topic_entity_for_template'] = template_id2topic_map[data['id']]
        for top1,top2 in zip(data['linked_topic_entity_for_human'],data['linked_topic_entity_for_template']):
            if top1 != top2:
                print(data['id'])

    with open('/data/lyt/exp/rush/kgqa/'+file+'.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

