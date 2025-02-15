from pandas import DataFrame
import pandas as pd
import torch
import torchhd
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
import os
import numpy as np
import random 
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import polars as pl

import time
from src.dictionary import *
from src.function import *


tqdm.pandas()
SEED = 42
# random.seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED) 
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False


# IF STOPWORDS NOT DOWNLOADED YET 
nltk.download('stopwords')



class FastVSS(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
        n_label: int,
        product_df: DataFrame,
        query_df: DataFrame,
        label_df: DataFrame,
        pretrain_w2v:str='',
        verbose:bool=True,
        pretrain_pvs:str='',
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        self.n_dimensions = n_dimensions
        self.n_label = n_label
        self.pdf = product_df
        self.qdf = query_df
        self.ldf = label_df
        self.verbose = verbose
        self.stop_words = set(stopwords.words('english'))
        self.dtype = dtype
        self.device = device
        self.pretrain_w2v=pretrain_w2v
        self.pretrain_pvs=pretrain_pvs
        
        #Initiate memory for concept query
        self.query = torchhd.embeddings.Random(num_embeddings=2,embedding_dim=self.n_dimensions,device=device)
        #initiate memory for concept product
        self.product_embedding = torchhd.embeddings.Random(num_embeddings=8,embedding_dim=self.n_dimensions,device=device)
        #Initiate memory for encode numeric feature into HV
        self.numeric = torchhd.embeddings.Level(num_embeddings=1000,embedding_dim=self.n_dimensions,low=0,high=1000,device=device)
        #Initiate associative memory for Label
        self.label = torchhd.random(num_vectors=self.n_label,dimensions=self.n_dimensions,device=device)

        if pretrain_w2v == '':
            status('Precomputing Dictionary',self.verbose)
            self.__init__dictionary()
        else:
            status('Loading Dictionary',self.verbose)
            self.w2v = Word2Vec.load(pretrain_w2v)

        if pretrain_pvs == '':
            status('Precomputing Product',self.verbose)
            self.__precompute_product()
        else:
            status('Precomputing Product',self.verbose)
            self.pvs = torch.load(pretrain_pvs)
            self.pdf[['rating_count', 'average_rating', 'review_count']] = self.pdf[['rating_count', 'average_rating', 'review_count']].fillna(0)
            self.pdf[['product_class', 'category hierarchy', 'product_description']] = self.pdf[['product_class', 'category hierarchy', 'product_description']].fillna('')

        status('Preparing WANDS',self.verbose)
        self.__prepare_wands()

        status('Building Model Done',self.verbose)
    
    def __init__dictionary(self)->None:
        tmp = self.pdf['product_description'].fillna('')+ ' ' + self.pdf['product_features'].fillna('')
        lemma = [lemmatize_sentence(clean_string(s)) for s in tqdm(tmp.values, desc="Lemmatizing and Cleaning Strings")]
        
        corpus = []
        for text in tqdm(lemma, desc="Tokenizing Text"):
            tokens = [word for word in text.lower().split(' ') if word.isalnum() and word not in self.stop_words]
            corpus.append(tokens)

        status('Building Dictionary',self.verbose)
        self.w2v = Word2Vec(vector_size=self.n_dimensions, window=5, min_count=1, workers=1, seed=SEED, epochs=100)
        self.w2v.build_vocab(corpus, desc="Building Vocabulary")
        self.w2v.train(corpus, total_examples=self.w2v.corpus_count, epochs=self.w2v.epochs)
        self.w2v.save(f"hyper_w2v_{self.n_dimensions}.model")
        status('Building Dictionary Done',self.verbose)
    
    def __compute_row(self,row)->torch.Tensor:
        vs = torch.stack([
             sen2v(row['product_name'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['product_class'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['category hierarchy'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['product_description'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['product_features'], self.w2v,self.n_dimensions,self.device),
             self.numeric(torch.tensor(row['rating_count'])),
             self.numeric(torch.tensor(row['average_rating'])), 
             self.numeric(torch.tensor(row['review_count'])),
        ]).mul(self.product_embedding.weight)
        return torchhd.multibundle(vs)


    def __precompute_product(self)->None:
        self.pdf[['rating_count', 'average_rating', 'review_count']] = self.pdf[['rating_count', 'average_rating', 'review_count']].fillna(0)
        self.pdf[['product_class', 'category hierarchy', 'product_description']] = self.pdf[['product_class', 'category hierarchy', 'product_description']].fillna('')
        self.pdf['pvs'] = self.pdf.apply(self.__compute_row,axis=1)
        self.pvs = torch.stack(list(self.pdf['pvs'].values))
        torch.save(self.pvs, f'hyper_pvs_{self.n_dimensions}.pt')
        status('Building Product Done',self.verbose)

    
    def __prepare_wands(self)->None:
        temp_ldf = self.ldf.copy()
        self.pdf['pvs'] = [torchhd.soft_quantize(self.pvs[i]) for i in range(self.pvs.shape[0])]
        temp_pdf = self.pdf[['product_id','pvs']]
        temp_ldf=temp_ldf.merge(self.qdf,how='left',left_on='query_id',right_on='query_id')
        temp_ldf=temp_ldf.merge(temp_pdf,how='left',left_on='product_id',right_on='product_id')

        cct = temp_ldf[['id','query_id','query','query_class','pvs','label']].fillna('')
        cct.loc[:, 'label'] = cct['label'].map(LABEL_ENCODE)
        cct['label'] = cct.label.astype(int)
        self.data = cct
    
    def fit(self,data:torch.Tensor,batch_size:int = 256)->None:
        train_data_np = data.to_numpy()
        num_samples = len(train_data_np)

        for i in range(0, num_samples, batch_size):
            batch = train_data_np[i:i + batch_size] 
            labels = batch[:, 5]
            queries = torch.stack([
                torch.stack(batch[:, 6].tolist()),
                torch.stack(batch[:, 4].tolist())
            ], dim=1).mul(self.query.weight)
            
            queries = torchhd.soft_quantize(torchhd.multibundle(queries))
            scores = torchhd.cosine_similarity(queries, self.label) 
            pred_inds = torch.argmax(scores, dim=1)  
            
            for idx in range(len(batch)):
                label = labels[idx]
                pred_ind = pred_inds[idx]
                query = queries[idx]
                alpha = scores[idx, pred_ind] - scores[idx, label] 
                
                if label != pred_ind:
                    self.label[pred_ind] = self.label[pred_ind].sub(query,alpha=alpha)
                    if scores[idx, label] <= 0.95:
                        self.label[label] = self.label[label].add(query,alpha=alpha)
                elif scores[idx, label] <= 0:
                    self.label[label] = self.label[label].add(query,alpha=scores[idx,label]*-1.)
                    
    def validate(self)->None:   
        skf = LeaveOneGroupOut()
        c = 0
        splits = skf.get_n_splits(groups=self.data['query_id'])

        query_vectors = self.data.groupby('query')['query'].apply(lambda q: sen2v(q.iloc[0], self.w2v, self.n_dimensions, self.device))
        self.data['query_vector'] = self.data['query'].map(query_vectors)
        mrl,ndcgl = [],[]
        with torch.no_grad():
            pbar = tqdm(skf.split(self.data[['query','query_class','pvs']],self.data['label'],self.data['query_id']), total=splits, desc=f"LOGOCV| FOLD {c+1}/{splits}")
            for train_index, val_index in pbar:
                self.label = torchhd.random(num_vectors=3,dimensions=self.n_dimensions,device=self.device)
                y_true_test,y_pred_test=[],[]
                
                train_data= self.data.iloc[train_index]
                train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
                self.fit(train_data,256)
        
                val_data =self.data.iloc[val_index]
                val_data_np = val_data.to_numpy()
                for row in range(len(val_data)):
                    pbar.set_description(f'LOGOCV| FOLD {c+1} | Test {row}/{len(val_data)}')  
                    label = val_data_np[row][5]
                    
                    query = torch.stack([
                            val_data_np[row][6],
                            val_data_np[row][4]
                        ]).mul(self.query.weight)
                    
                    query = torchhd.soft_quantize(torchhd.multibundle(query))
                    scores = torchhd.cosine_similarity(query, self.label)
                    pred_ind = int(torch.argmax(scores))

                    y_true_test.append(label)
                    y_pred_test.append(pred_ind)

                        
                tmp = pd.DataFrame({'true': [SCORE[i] for i in y_true_test],'pred': [SCORE[i] for i in y_pred_test],'conf':y_pred_test}).sort_values(by=['pred','conf'], ascending=False)
                ndcgl.append(ndcg(tmp['true'],10))
                mrl.append(mrr(tmp))
                pbar.set_postfix(
                    ndcg_10 = f'{np.sum(ndcgl)/len(ndcgl):.2f}',
                    mrr = f'{np.sum(mrl)/len(mrl):.2f}'
                    )
                c+=1


  