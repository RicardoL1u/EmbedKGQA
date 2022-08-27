import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from tqdm import tqdm
import argparse
import pickle
import operator
from torch.nn import functional as F
from dataloader import DatasetMetaQA, DataLoaderMetaQA
from model import RelationExtractor
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import networkx as nx
from utils import *
import time
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
from kge.model import KgeModel
from kge.util.io import load_checkpoint

parser = argparse.ArgumentParser()


parser.add_argument('--hops', type=str, default='1')
parser.add_argument('--load_from', type=str, default='')
parser.add_argument('--ls', type=float, default=0.0)
parser.add_argument('--validate_every', type=int, default=5)
parser.add_argument('--model', type=str, default='ComplEx')
parser.add_argument('--mode', type=str, default='eval')
parser.add_argument('--outfile', type=str, default='best_score_model')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--entdrop', type=float, default=0.0)
parser.add_argument('--reldrop', type=float, default=0.0)
parser.add_argument('--scoredrop', type=float, default=0.0)
parser.add_argument('--l3_reg', type=float, default=0.0)
parser.add_argument('--decay', type=float, default=1.0)
parser.add_argument('--shuffle_data', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=90)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--neg_batch_size', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=200)
parser.add_argument('--embedding_dim', type=int, default=256)
parser.add_argument('--relation_dim', type=int, default=30)
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--freeze', type=str2bool, default=True)
parser.add_argument('--do_batch_norm', type=str2bool, default=True)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
args = parser.parse_args()


def train(data_path, neg_batch_size, batch_size, shuffle, num_workers, nb_epochs, embedding_dim, hidden_dim, relation_dim, gpu, use_cuda,patience, freeze, validate_every, hops, lr, entdrop, reldrop, scoredrop, l3_reg, model_name, decay, ls, load_from, outfile, do_batch_norm, valid_data_path=None):
    print('Loading entities and relations')
    entity2idx, idx2entity, embedding_matrix = None,None,None
    if 'wikidata5m' in hops:
        with open("complex_wikidata5m.pkl", "rb") as fin:
            wiki5m = pickle.load(fin)
        entity2idx = wiki5m.graph.entity2id
        idx2entity = {v:k for k,v in  entity2idx.items()}
        embedding_matrix = torch.tensor(wiki5m.solver.entity_embeddings)
    else:
        kg_type = 'full'
        if 'half' in hops:
            kg_type = 'half'
        checkpoint_file = '../../pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
        print('Loading kg embeddings from', checkpoint_file)
        kge_checkpoint = load_checkpoint(checkpoint_file)
        kge_model = KgeModel.create_from(kge_checkpoint)
        kge_model.eval()
        e = getEntityEmbeddings(kge_model, hops)
        entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)

    print('Loaded entities and relations')

    data = process_text_file(data_path, split=False)
    print('Train file processed, making dataloader')
    # word2ix,idx2word, max_len = get_vocab(data)
    # hops = str(num_hops)
    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, entity2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, pretrained_embeddings=embedding_matrix, freeze=freeze, device=device, entdrop = entdrop, reldrop = reldrop, scoredrop = scoredrop, l3_reg = l3_reg, model = model_name, ls = ls, do_batch_norm=do_batch_norm)
    print('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = "checkpoints/roberta_finetune/" + load_from + ".pt"
        # fname = "/scratche/home/apoorv/tut_pytorch/kg-qa/checkpoints/roberta_finetune/" + load_from + ".pt"
        model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, decay)
    optimizer.zero_grad()
    best_score = -float("inf")
    best_model = model.state_dict()
    no_update = 0
    # time.sleep(10)
    for epoch in range(nb_epochs):
        phases = []
        for i in range(validate_every):
            phases.append('train')
        phases.append('valid')
        for phase in phases:
            if phase == 'train':
                model.train()
                # model.apply(set_bn_eval)
                loader = tqdm(data_loader, total=len(data_loader), unit="batches")
                running_loss = 0
                for i_batch, a in enumerate(loader):
                    model.zero_grad()
                    question_tokenized = a[0].to(device)
                    attention_mask = a[1].to(device)
                    positive_head = a[2].to(device)
                    positive_tail = a[3].to(device) 
                    loss = model(question_tokenized=question_tokenized, attention_mask=attention_mask, p_head=positive_head, p_tail=positive_tail)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    loader.set_postfix(Loss=running_loss/((i_batch+1)*batch_size), Epoch=epoch)
                    loader.set_description('{}/{}'.format(epoch, nb_epochs))
                    loader.update()
                
                scheduler.step()

            elif phase=='valid':
                model.eval()
                eps = 0.0001
                answers, score = validate_v2(model=model, data_path= valid_data_path, entity2idx=entity2idx, dataloader=dataset, device=device, model_name=model_name)
                if score > best_score + eps:
                    best_score = score
                    no_update = 0
                    best_model = model.state_dict()
                    print(hops + " hop Validation accuracy (no relation scoring) increased from previous epoch", score)
                    # writeToFile(answers, 'results_' + model_name + '_' + hops + '.txt')
                    torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                elif (score < best_score + eps) and (no_update < patience):
                    no_update +=1
                    print("Validation accuracy decreases to %f from %f, %d more epoch to check"%(score, best_score, patience-no_update))
                elif no_update == patience:
                    print("Model has exceed patience. Saving best model and exiting")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                if epoch == nb_epochs-1:
                    print("Final Epoch has reached. Stoping and saving model.")
                    # torch.save(best_model, "checkpoints/roberta_finetune/best_score_model.pt")
                    # torch.save(best_model, "checkpoints/roberta_finetune/" + outfile + ".pt")
                    exit()
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/"+str(epoch)+".pt")
                # torch.save(model.state_dict(), "checkpoints/roberta_finetune/x.pt")

def eval(data_path,
    load_from,
    gpu,
    hidden_dim,
    relation_dim,
    embedding_dim,
    hops,
    batch_size,
    num_workers,
    model_name,
    do_batch_norm,
    use_cuda):

    print('Loading entities and relations')
    entity2idx, idx2entity, embedding_matrix = None,None,None
    if 'wikidata5m' in hops:
        with open("complex_wikidata5m.pkl", "rb") as fin:
            wiki5m = pickle.load(fin)
        entity2idx = wiki5m.graph.entity2id
        idx2entity = {v:k for k,v in  entity2idx.items()}
        embedding_matrix = torch.tensor(wiki5m.solver.entity_embeddings)
    else:
        kg_type = 'full'
        if 'half' in hops:
            kg_type = 'half'
        checkpoint_file = '../../pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
        print('Loading kg embeddings from', checkpoint_file)
        kge_checkpoint = load_checkpoint(checkpoint_file)
        kge_model = KgeModel.create_from(kge_checkpoint)
        kge_model.eval()
        e = getEntityEmbeddings(kge_model, hops)
        entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)

    print('Loaded entities and relations')

    entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
    data = process_text_file(data_path, split=False)
    print('Evaluation file processed, making dataloader')

    device = torch.device(gpu if use_cuda else "cpu")
    dataset = DatasetMetaQA(data, entity2idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    print('Creating model...')
    model = RelationExtractor(embedding_dim=embedding_dim, num_entities = len(idx2entity), relation_dim=relation_dim, 
                              pretrained_embeddings=embedding_matrix, device=device, 
                              model = model_name, do_batch_norm=do_batch_norm)
    print('Model created!')
    if load_from != '':
        # model.load_state_dict(torch.load("checkpoints/roberta_finetune/" + load_from + ".pt"))
        fname = "checkpoints/roberta_finetune/" + load_from + ".pt"
        print('Loading from %s' % fname)
        model.load_state_dict(torch.load(fname, map_location=torch.device('cpu')))
        print('Loaded successfully!')
    else:
        print('Need to specify load_from argument for evaluation!')
        exit(0)
    
    model.to(device)
    answers, score = validate_v2(model=model, data_path= data_path, 
                                 entity2idx=entity2idx, dataloader=dataset, 
                                 device=device, model_name=model_name,
                                 writeCandidatesToFile=True)
    print('Score', score)





hops = args.hops

model_name = args.model

if 'webqsp' in hops:
    data_path = '../../data/QA_data/WebQuestionsSP/qa_train_webqsp.txt'
    valid_data_path = '../../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt'
    test_data_path = '../../data/QA_data/WebQuestionsSP/qa_test_webqsp.txt'

print(f'the args is')
print(args)

if args.mode == 'train':
    print(args)
    train(data_path=data_path, 
    neg_batch_size=args.neg_batch_size, 
    batch_size=args.batch_size,
    shuffle=args.shuffle_data, 
    num_workers=args.num_workers,
    nb_epochs=args.nb_epochs, 
    embedding_dim=args.embedding_dim, 
    hidden_dim=args.hidden_dim, 
    relation_dim=args.relation_dim, 
    gpu=args.gpu, 
    use_cuda=args.use_cuda, 
    valid_data_path=valid_data_path,
    patience=args.patience,
    validate_every=args.validate_every,
    freeze=args.freeze,
    hops=args.hops,
    lr=args.lr,
    entdrop=args.entdrop,
    reldrop=args.reldrop,
    scoredrop = args.scoredrop,
    l3_reg = args.l3_reg,
    model_name=args.model,
    decay=args.decay,
    ls=args.ls,
    load_from=args.load_from,
    outfile=args.outfile,
    do_batch_norm=args.do_batch_norm)



elif args.mode == 'eval':
    eval(data_path = test_data_path,
    load_from=args.load_from,
    gpu=args.gpu,
    hidden_dim=args.hidden_dim,
    relation_dim=args.relation_dim,
    embedding_dim=args.embedding_dim,
    hops=args.hops,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    model_name=args.model,
    do_batch_norm=args.do_batch_norm,
    use_cuda=args.use_cuda)
