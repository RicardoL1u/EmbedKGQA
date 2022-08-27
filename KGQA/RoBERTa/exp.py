# %%
from transformers import RobertaModel,RobertaTokenizer
from utils import *
import torch
# %%
device = torch.device(6)


# %%
roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# %%
import sys
sys.path.append('../..')
from kge.model import KgeModel
from kge.util.io import load_checkpoint


# %%
kg_type = 'half'
hops = 'webqsp_half'
checkpoint_file = '../../pretrained_models/embeddings/ComplEx_fbwq_' + kg_type + '/checkpoint_best.pt'
kge_checkpoint = load_checkpoint(checkpoint_file)
kge_model = KgeModel.create_from(kge_checkpoint)
kge_model.eval()
e = getEntityEmbeddings(kge_model, hops)

# %%
from torch.utils.data import Dataset, DataLoader
from dataloader import DatasetMetaQA, DataLoaderMetaQA


# %%


# %%
data_path = '../../data/QA_data/WebQuestionsSP/qa_train_webqsp.txt'
print(type(e))
print(list(e.keys())[:30])
entity2idx, idx2entity, embedding_matrix = prepare_embeddings(e)
print(type(entity2idx))
print(type(idx2entity))
print(type(embedding_matrix))
print(type(embedding_matrix[0]))
print(embedding_matrix[0].shape)
data = process_text_file(data_path, split=False)
print('Train file processed, making dataloader')
# word2ix,idx2word, max_len = get_vocab(data)
# hops = str(num_hops)
device = torch.device(device)
dataset = DatasetMetaQA(data, e, entity2idx)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

# %%
a = next(iter(data_loader))

# %%
question_tokenized = a[0].to(device)
attention_mask = a[1].to(device)
positive_head = a[2].to(device)
positive_tail = a[3].to(device)  

# %%
roberta_tokenizer.decode(question_tokenized[0].tolist(),skip_special_tokens=True)

# %%
question_tokenized[0]

# %%
test = roberta_tokenizer.encode('There is a test')
roberta_tokenizer.decode(test)

# %%


# %%
device


# %%
question_tokenized

# %%
roberta_model(question_tokenized, attention_mask=attention_mask)

# %%



