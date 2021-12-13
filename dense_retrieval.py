import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np
import os.path
from os import path
import pandas as pd
import pickle
import re
import urllib.request
import requests 
import shutil 

def generate_id_text():
  if not path.exists("ID-TEXT.csv"):
    df = pd.read_csv('met_description.csv')
    df.drop(['Dimensions',"Gallery Number",'Unnamed: 0','Object Number','Is Highlight','Is Timeline Work','Is Public Domain','Rights and Reproduction','Link Resource','Object Wikidata URL','Tags AAT URL','Tags Wikidata URL'], axis = 1,inplace=True)
    df['Artist Display Name'] = df['Artist Display Name'].astype(str) + " "+df['Artist Display Name'].astype(str)
    df['Title'] = df['Title'].astype(str) + " "+df['Title'].astype(str)
    df['text'] = df[df.columns[1:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )
    new_df = df[['Object ID','text']].copy()
    new_df.to_csv('ID-TEXT.csv', index=False)
  else:
    new_df=pd.read_csv('ID-TEXT.csv')
    ID=new_df['Object ID'].to_list()
    TEXT=new_df['text'].to_list()
  return (ID,TEXT)


def tokenizer(TEXT):
  Tokens=[]
  for text in TEXT:
    tokens=[]
    for token in text.lower().split():
      token = token.strip(string.punctuation)
      if token not in _stop_words.ENGLISH_STOP_WORDS:
        tokens.append(token)
    Tokens.append(tokens)
  return Tokens

def bm25_result(Tokens,query):
  url="https://www.metmuseum.org/art/collection/search/"
  bm25 = BM25Okapi(Tokens)
  q=[]
  for terms in query.lower().split():
    t=terms.strip(string.punctuation)
    if t not in _stop_words.ENGLISH_STOP_WORDS:
      q.append(t)
  bm25_scores = bm25.get_scores(q)
  top_n = np.argpartition(bm25_scores, -5)[-5:]
  bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
  bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
  print("Top5 BM25 Retrieval:")
  result=[]
  for hit in bm25_hits[0:5]:
      print("\t{:.3f}\t{}".format(hit['score'], url+str((ID[hit['corpus_id']]))))
      result.append((ID[hit['corpus_id']]))
  return result

def get_text_emb(bi_encoder):
  if path.exists("embeddings_text"):
    embeddings = pickle.load( open( "embeddings_text", "rb" ) )
  else:
    ID,TEXT=generate_id_text()
    embeddings=[]
    for i in tqdm(range(len(TEXT))):
      context=TEXT[i]
      embedding = bi_encoder.encode(context, convert_to_tensor=True)
      if i==0:
        embeddings=embedding
      else:
        embeddings=torch.vstack((embeddings,embedding))
    pickle.dump( embeddings, open( "embeddings_text", "wb" ) )
    embeddings = pickle.load( open( "embeddings_text", "rb" ) )
  return embeddings

def semnatic_search(query,bi_encoder,cross_encoder,top_k,ID,TEXT):
  url="https://www.metmuseum.org/art/collection/search/"
  question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
  question_embedding = question_embedding.cuda()
  corpus_embeddings=get_text_emb(bi_encoder).cuda()

  hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)[0]
  cross_corpus = [[query, TEXT[hit['corpus_id']]] for hit in hits]
  cross_scores = cross_encoder.predict(cross_corpus)
  for i in range(len(cross_scores)):
      hits[i]['cross-score'] = cross_scores[i]
  
  Tokens=tokenizer(TEXT)
  r1=bm25_result(Tokens,query)

  print("Top5 Bi-Encoder Retrieval")
  hits = sorted(hits, key=lambda x: x['score'], reverse=True)
  r2=[]
  for hit in hits[0:5]:
      print("\t{:.3f}\t{}".format(hit['score'], url+str(ID[hit['corpus_id']])))
      r2.append((ID[hit['corpus_id']]))
  r3=[]
  print("Top5 Cross-Encoder Re-ranker Retrieval")
  hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
  for hit in hits[0:5]:
      print("\t{:.3f}\t{}".format(hit['cross-score'], url+str(ID[hit['corpus_id']])))
      r3.append((ID[hit['corpus_id']]))
  return [r1,r2,r3]

def run(query,bi_encoder,cross_encoder,top_k,eval):
  if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")
  ID,TEXT=generate_id_text()
  Tokens=tokenizer(TEXT)
  r1,r2,r3=semnatic_search(query,bi_encoder,cross_encoder,top_k,ID,TEXT)
  if eval==True:
    evaluate(r1,r2,r3)

def evaluate(r1,r2,r3):
  topics_df=pd.read_csv('topics.csv')
  topics=topics_df["query"].to_list()
  qrel_df=pd.read_csv("qrels.csv")

  qrels=[]
  docno=[]
  for i in range(len(topics)):
    q=qrel_df["label"].to_list()[100*i:100*i+100]
    qrels.append(q)
    docno.append(qrel_df["docno"].to_list()[100*i:100*i+100])
  m=0
  rel1=[]
  rel2=[]
  rel3=[]
  for i in range(len(r1)):
    if r1[i] in docno[m]:
      index1 = docno[m].index(r1[i])
      rel1.append(qrels[m][index1])
    else: rel1.append(3)
    if r2[i] in docno[m]:
      index2 = docno[m].index(r2[i])
      rel2.append(qrels[m][index2])
    else: rel2.append(3)
    if r3[i] in docno[m]:
      index3 = docno[m].index(r3[i])
      rel3.append(qrels[m][index3])
    else: rel3.append(3)
  dcg1=rel1[0]
  dcg2=rel2[0]
  dcg3=rel3[0]
  for i in range(1,len(r1)):
    dcg1+=rel1[i]/np.log2(i+1)
    dcg2+=rel2[i]/np.log2(i+1)
    dcg3+=rel3[i]/np.log2(i+1)

  true1=sorted(qrels[m], reverse=True)
  true2=sorted(qrels[m], reverse=True)
  true3=sorted(qrels[m], reverse=True)
  true_dcg1=true1[0]
  true_dcg2=true2[0]
  true_dcg3=true3[0]
  for i in range(1,len(r1)):
    true_dcg1+=true1[i]/np.log2(i+1)
    true_dcg2+=true2[i]/np.log2(i+1)
    true_dcg3+=true3[i]/np.log2(i+1)

  ndcg1=dcg1/true_dcg1
  ndcg2=dcg2/true_dcg2
  ndcg3=dcg3/true_dcg3
  print("The NDCG@5 of BM25 is: ",ndcg1)
  print("The NDCG@5 of Bi-encoder is: ",ndcg2)
  print("The NDCG@5 of Cross-Encoder Re-ranker is: ",ndcg3)

def extract_image(id):
  with urllib.request.urlopen("https://www.metmuseum.org/art/collection/search/"+id) as url:
    s = url.read()  
  s.decode('UTF-8')
  s=str(s).replace('\\n',"").replace('\\r',"")
  x = re.search("https://collectionapi.metmuseum.*\W", s)
  x=x.group(0)
  startIndex = x.find('\"')
  x=x[0:startIndex]
  image_url = x
  filename = id+".jpg"
  r = requests.get(image_url, stream = True)
  if r.status_code == 200:
      r.raw.decode_content = True
      with open(filename,'wb') as f:
          shutil.copyfileobj(r.raw, f)      
      print('Image sucessfully Downloaded: ',filename)
  else:
      print('Image Couldn\'t be retreived')

if __name__ == '__main__':
    bi_encoder = SentenceTransformer('msmarco-distilbert-base-v2')
    bi_encoder.max_seq_length = 512
    top_k = 100
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    ID,TEXT=generate_id_text()
    Tokens=tokenizer(TEXT)
    print('Enter the query:')
    x = input()
    eval=False
    run( x,bi_encoder,cross_encoder,100,eval)