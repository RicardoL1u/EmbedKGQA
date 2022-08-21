# KGQA

## Single-Hop Questoin Answering

## Multi-Hop Question Answering
we focus on this 


# KG Embeddings

Task: For each entity e and relation r in KG define embedding vector $e_e$, $e_r$
Find a funtion $\phi(h,r,t) = f(e_h,e_r,e_t)$ such that 
$ \phi(h,r,t) > 0 $ if (h,r,t) is an edge in KG else no edge in KG
# Challenge

## Challenge 1 KG incompleteness
**KGs are incomplete** 在推理链路中的某一跳可能不存在于KG中

May cause breaking of reasoning chain

Solution: use additional data e.g: text corpus

> we assume there are no text corpus

## Challenge 2 Neighbourhood Limitations

Millions of enetitis in the KG

we usually assume **that the answer may appear in the n-hop neighborhood**

but if the answer may **still out of scope of the n-hop neighborhood** since the **KG is incomplete**

> more than 3 hop not feasible -> too bigger

# Problem Statement
Training data: KG + (NL question, topic entity, answer entities)

**NO Path Annotation**

> eg:
> Q: What are the genres of movies direct by Louis Meills?
> Topic Entity: Louis Mellis
> Answer Entity: Crime

# Our Solution
EmbedKGQA first embeds the entities of hte KG in a vector space
QA performed using these embeddings
This way EmbedKGQA can
1. Deal with KG incompleteness -> since we can implicity take the advantage of KG embeddings to predict there is one edge between 2 edges or non
2. No Neighbourhodd limitations 
