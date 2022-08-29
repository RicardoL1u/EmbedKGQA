# import os
import torch
import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn
import pickle
from tqdm import tqdm
# import argparse
# import operator
from torch.nn import functional as F
# from dataloader import DatasetMetaQA, DataLoaderMetaQA
# from model import RelationExtractor
# from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
# import time
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
# from kge.model import KgeModel
# from kge.util.io import load_checkpoint

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True

def prepare_embeddings(embedding_dict):
    entity2idx = {}
    idx2entity = {}
    i = 0
    embedding_matrix = []
    for key, entity in embedding_dict.items():
        entity2idx[key] = i
        idx2entity[i] = key
        i += 1
        embedding_matrix.append(entity)
    return entity2idx, idx2entity, embedding_matrix

def get_vocab(data):
    word_to_ix = {}
    maxLength = 0
    idx2word = {}
    for d in data:
            sent = d[1]
            for word in sent.split():
                if word not in word_to_ix:
                    idx2word[len(word_to_ix)] = word
                    word_to_ix[word] = len(word_to_ix)
                    
            length = len(sent.split())
            if length > maxLength:
                maxLength = length

    return word_to_ix, idx2word, maxLength

def preprocess_entities_relations(entity_dict, relation_dict, entities, relations):
    e = {}
    r = {}
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = entities[ent_id]
    f.close()
    f = open(relation_dict,'r')
    for line in f:
        line = line.strip().split('\t')
        rel_id = int(line[0])
        rel_name = line[1]
        r[rel_name] = relations[rel_id]
    f.close()
    return e,r

def makeGraph(entity2idx):
    f = open('kb.txt', 'r')
    triples = []
    for line in f:
        line = line.strip().split('|')
        triples.append(line)
    f.close()
    G = nx.Graph()
    for t in triples:
        e1 = entity2idx[t[0]]
        e2 = entity2idx[t[2]]
        G.add_node(e1)
        G.add_node(e2)
        G.add_edge(e1, e2)
    return G

def getBest(scores, candidates):
    cand_scores_dict = {}
    highest = 0
    highest_key = ''
    for c in candidates:
        if scores[c] > highest:
            highest = scores[c]
            highest_key = c
    return highest_key
    

def getNeighbourhood(graph, entity, radius=1):
    g = nx.ego_graph(graph, entity, radius, center=False)
    nodes = list(g.nodes)
    return nodes


def getMask(candidates, entity2idx):
    max_len = len(entity2idx)
    x = np.ones(max_len)
    for c in candidates:
        if c not in entity2idx:
            c = c.strip()
        x[entity2idx[c]] = 0
    return x

def inTopk(scores, ans, k):
    result = False
    topk = torch.topk(scores, k)[1]
    for x in topk:
        if x in ans:
            result = True
    return result

def validate(dataset,model,device,writeCandidatesToFile=False):
    hit_at_10 = 0
    answers = []
    candidates_with_scores = []
    num_incorrect = 0
    total_correct = 0
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    for d in tqdm(dataset):
        # try:
        question_tokenized = d[0].to(device)
        attention_mask = d[1].to(device)
        head = d[2].to(device)
        ans = d[3].to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        # candidates = qa_nbhood_list[i]
        # mask = torch.from_numpy(getMask(candidates, entity2idx)).to(device)
        # following 2 lines for no neighbourhood check
        mask = torch.zeros(len(dataset.entity2idx)).to(device)
        mask[head] = 1
        #reduce scores of all non-candidates
        new_scores = scores - (mask*99999)
        pred_ans = torch.argmax(new_scores).item()
        # new_scores = new_scores.cpu().detach().numpy()
        # scores_list.append(new_scores)
        if pred_ans == head.item():
            print('Head and answer same')
            print(torch.max(new_scores))
            print(torch.min(new_scores))
        # pred_ans = getBest(scores, candidates)
        # if ans[0] not in candidates:
        #     print('Answer not in candidates')
            # print(len(candidates))
            # exit(0)
        
        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = dataset.idx2entity[head.item()]
            entry['head'] = head_text
            s, c =  torch.topk(new_scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(dataset.idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(dataset.idx2entity[a.item()])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)


        if inTopk(new_scores, ans, 10):
            hit_at_10 += 1

        if type(ans) is int:
            ans = [ans]
        is_correct = 0
        if pred_ans in ans:
            total_correct += 1
            is_correct = 1
        else:
            num_incorrect += 1
        q_text = d[0]
        answers.append(dataset.tokenizer.decode(q_text) + '\t' + str(pred_ans) + '\t' + str(is_correct))
        # except:
        #     error_count += 1
        
    if writeCandidatesToFile:
        # pickle.dump(candidates_with_scores, open('candidates_with_score_and_qe_half.pkl', 'wb'))
        pickle.dump(candidates_with_scores, open('webqsp_scores_full_kg_fixed.pkl', 'wb'))
        print('wrote candidate file (for future answer processing)')
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    # print(hit_at_10/len(data))
    accuracy = total_correct/len(dataset)
    # print('Error mean rank: %f' % (incorrect_rank_sum/num_incorrect))
    # print('%d out of %d incorrect were not in top 50' % (not_in_top_50_count, num_incorrect))
    return answers, accuracy

def validate_v2(data_path, device, model, dataloader, entity2idx, model_name, writeCandidatesToFile=False):
    model.eval()
    data = process_text_file(data_path)
    idx2entity = {}
    for key, value in entity2idx.items():
        idx2entity[value] = key
    answers = []
    data_gen = data_generator(data=data, dataloader=dataloader, entity2idx=entity2idx)
    total_correct = 0
    error_count = 0
    num_incorrect = 0
    incorrect_rank_sum = 0
    not_in_top_50_count = 0

    # print('Loading nbhood file')
    # if 'webqsp' in data_path:
    #     # with open('webqsp_test_candidate_list_only2hop.pickle', 'rb') as handle:
    #     with open('webqsp_test_candidate_list_half.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)
    # else:
    #     with open('qa_dev_full_candidate_list.pickle', 'rb') as handle:
    #         qa_nbhood_list = pickle.load(handle)

    scores_list = []
    hit_at_10 = 0
    candidates_with_scores = []
    for i in tqdm(range(len(data))):
        # try:
        d = next(data_gen)
        head = d[0].to(device)
        question_tokenized = d[1].to(device)
        attention_mask = d[2].to(device)
        ans = d[3]
        tail_test = torch.tensor(ans, dtype=torch.long).to(device)
        scores = model.get_score_ranked(head=head, question_tokenized=question_tokenized, attention_mask=attention_mask)[0]
        # candidates = qa_nbhood_list[i]
        # mask = torch.from_numpy(getMask(candidates, entity2idx)).to(device)
        # following 2 lines for no neighbourhood check
        mask = torch.zeros(len(entity2idx)).to(device)
        mask[head] = 1
        #reduce scores of all non-candidates
        new_scores = scores - (mask*99999)
        pred_ans = torch.argmax(new_scores).item()
        # new_scores = new_scores.cpu().detach().numpy()
        # scores_list.append(new_scores)
        if pred_ans == head.item():
            print('Head and answer same')
            print(torch.max(new_scores))
            print(torch.min(new_scores))
        # pred_ans = getBest(scores, candidates)
        # if ans[0] not in candidates:
        #     print('Answer not in candidates')
            # print(len(candidates))
            # exit(0)
        
        if writeCandidatesToFile:
            entry = {}
            entry['question'] = d[-1]
            head_text = idx2entity[head.item()]
            entry['head'] = head_text
            s, c =  torch.topk(new_scores, 200)
            s = s.cpu().detach().numpy()
            c = c.cpu().detach().numpy()
            cands = []
            for cand in c:
                cands.append(idx2entity[cand])
            entry['scores'] = s
            entry['candidates'] = cands
            correct_ans = []
            for a in ans:
                correct_ans.append(idx2entity[a])
            entry['answers'] = correct_ans
            candidates_with_scores.append(entry)


        if inTopk(new_scores, ans, 10):
            hit_at_10 += 1

        if type(ans) is int:
            ans = [ans]
        is_correct = 0
        if pred_ans in ans:
            total_correct += 1
            is_correct = 1
        else:
            num_incorrect += 1
        q_text = d[-1]
        answers.append(q_text + '\t' + str(pred_ans) + '\t' + str(is_correct))
        # except:
        #     error_count += 1
        
    if writeCandidatesToFile:
        # pickle.dump(candidates_with_scores, open('candidates_with_score_and_qe_half.pkl', 'wb'))
        pickle.dump(candidates_with_scores, open('webqsp_scores_full_kg_fixed.pkl', 'wb'))
        print('wrote candidate file (for future answer processing)')
    # np.save("scores_webqsp_complex.npy", scores_list)
    # exit(0)
    # print(hit_at_10/len(data))
    accuracy = total_correct/len(data)
    # print('Error mean rank: %f' % (incorrect_rank_sum/num_incorrect))
    # print('%d out of %d incorrect were not in top 50' % (not_in_top_50_count, num_incorrect))
    return answers, accuracy

def writeToFile(lines, fname):
    f = open(fname, 'w')
    for line in lines:
        f.write(line + '\n')
    f.close()
    print('Wrote to ', fname)
    return

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm1d') != -1:
        m.eval()

def getEntityEmbeddings(kge_model, hops):
    e = {}
    entity_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_full/entity_ids.del'
    if 'half' in hops:
        entity_dict = '../../pretrained_models/embeddings/ComplEx_fbwq_half/entity_ids.del'
        print('Loading half entity_ids.del')
    embedder = kge_model._entity_embedder
    f = open(entity_dict, 'r')
    for line in f:
        line = line[:-1].split('\t')
        ent_id = int(line[0])
        ent_name = line[1]
        e[ent_name] = embedder._embeddings(torch.LongTensor([ent_id]))[0]
    f.close()
    return e

def process_text_file(text_file, split=False):
    data_file = open(text_file, 'r')
    data_array = []
    for data_line in data_file.readlines():
        data_line = data_line.strip()
        if data_line == '':
            continue
        data_line = data_line.strip().split('\t')
        # if no answer
        if len(data_line) != 2:
            continue
        question = data_line[0].split('[')
        question_1 = question[0]
        question_2 = question[1].split(']')
        head = question_2[0].strip()
        question_2 = question_2[1]
        question = question_1+'NE'+question_2
        ans = data_line[1].split('|')
        data_array.append([head, question.strip(), ans])
    if split==False:
        return data_array
    else:
        data = []
        for line in data_array:
            head = line[0]
            question = line[1]
            tails = line[2]
            for tail in tails:
                data.append([head, question, tail])
        return data

def data_generator(data, dataloader, entity2idx):
    for i in range(len(data)):
        data_sample = data[i]
        head = entity2idx[data_sample[0].strip()]
        question = data_sample[1]
        question_tokenized, attention_mask = dataloader.tokenize_question(question)
        if type(data_sample[2]) is str:
            ans = entity2idx[data_sample[2]]
        else:
            #TODO: not sure if this is the right way
            ans = []
            for entity in list(data_sample[2]):
                if entity.strip() in entity2idx:
                    ans.append(entity2idx[entity.strip()])
            # ans = [entity2idx[entity.strip()] for entity in list(data_sample[2])]

        yield torch.tensor(head, dtype=torch.long), question_tokenized, attention_mask, ans, data_sample[1]


