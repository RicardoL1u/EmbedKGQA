from transformers import RobertaModel,RobertaTokenizer
from utils import *
import torch
import pickle
import json
from dataloader import *

batch_size = 2
num_workers = 2
hops = 'wikidata5m'
data_path = '/data/lyt/exp/EmbedKGQA/train.json'


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
special_tokens = ['<spt>']
tokenizer.add_tokens(special_tokens,special_tokens=True)

print('Loading entities and relations')
entity2idx, idx2entity, embedding_matrix = None,None,None
if 'wikidata5m' in hops:
    with open("complex_wikidata5m.pkl", "rb") as fin:
        wiki5m = pickle.load(fin)
    entity2idx = wiki5m.graph.entity2id
    idx2entity = {v:k for k,v in  entity2idx.items()}
    embedding_matrix = torch.tensor(wiki5m.solver.entity_embeddings)
else:
    pass
    # kg_type = 'full'
    # if 'half' in hops:
    #     kg_type = 'half'
    # checkpoint_file = '../../pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
    # print('Loading kg embeddings from', checkpoint_file)
    # kge_checkpoint = load_checkpoint(checkpoint_file)
    # kge_model = KgeModel.create_from(kge_checkpoint)
    # kge_model.eval()
    # e = getEntityEmbeddings(kge_model, hops)
    # entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)

print('Loaded entities and relations')
device = torch.device(0)

    
print('Train file processed, making dataloader')
dataset = DatasetMetaQA(process_text_file(data_path, split=False), entity2idx, tokenizer) if '5m' not in hops \
    else DatasetAnonyQA(json.load(open(data_path)), entity2idx, tokenizer)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

a = next(iter(data_loader))

question_tokenized = a[0].to(device)
attention_mask = a[1].to(device)
positive_head = a[2].to(device)
positive_tail = a[3].to(device)

model = RobertaModel.from_pretrained('roberta-base').to(device)

if len(special_tokens) > 0:
    model.resize_token_embeddings(len(tokenizer))
print(tokenizer.batch_decode(question_tokenized,skip_special_tokens=True))
print(model(question_tokenized,attention_mask=attention_mask))
