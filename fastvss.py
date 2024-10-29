from typing import Optional, Literal, Callable, Iterable, Tuple
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
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import polars as pl

import time
from dictionary import *
from function import *


tqdm.pandas()
SEED = 42
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False


# IF STOPWORDS NOT DOWNLOADED YET 
nltk.download('stopwords')



class FastVSS(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
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
        self.pdf = product_df
        self.qdf = query_df
        self.ldf = label_df
        self.verbose = verbose
        
        self.stop_words = set(stopwords.words('english'))
        self.dtype = dtype
        self.device = device
        self.pretrain_w2v=pretrain_w2v
        self.pretrain_pvs=pretrain_pvs

        self.query = torchhd.embeddings.Random(num_embeddings=3,embedding_dim=self.n_dimensions,device=device)
        
        self.pro_embed = torchhd.embeddings.Random(num_embeddings=8,embedding_dim=self.n_dimensions,device=device)
        self.conti = torchhd.embeddings.Level(num_embeddings=1000,embedding_dim=self.n_dimensions,low=0,high=1000,device=device)

        self.label = torchhd.random(num_vectors=3,dimensions=self.n_dimensions,device=device)

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

        self.__prepare_qclass()
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
        self.w2v.build_vocab(tqdm(corpus, desc="Building Vocabulary"))
        self.w2v.train(tqdm(corpus, desc="Training Word2Vec"), total_examples=self.w2v.corpus_count, epochs=self.w2v.epochs)
        self.w2v.save(f"hyper_w2v_{self.n_dimensions}.model")
        status('Building Dictionary Done',self.verbose)
    
    
    def __compute_row(self,row)->torch.Tensor:
        vs = torch.stack([
             sen2v(row['product_name'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['product_class'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['category hierarchy'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['product_description'], self.w2v,self.n_dimensions,self.device),
             sen2v(row['product_features'], self.w2v,self.n_dimensions,self.device),
             self.conti(torch.tensor(row['rating_count'])),
             self.conti(torch.tensor(row['average_rating'])), 
             self.conti(torch.tensor(row['review_count'])),
        ]).mul(self.pro_embed.weight)
        return torchhd.multibundle(vs)


    def __precompute_product(self)->None:
        self.pdf[['rating_count', 'average_rating', 'review_count']] = self.pdf[['rating_count', 'average_rating', 'review_count']].fillna(0)
        self.pdf[['product_class', 'category hierarchy', 'product_description']] = self.pdf[['product_class', 'category hierarchy', 'product_description']].fillna('')
        self.pdf['pvs'] = self.pdf.apply(self.__compute_row,axis=1)
        self.pvs = torch.stack(list(self.pdf['pvs'].values))
        torch.save(self.pvs, f'hyper_pvs_{self.n_dimensions}.pt')
        status('Building Product Done',self.verbose)

    def __prepare_qclass(self)->None:
        self.qclass_dict = {}
        columns = list(self.pdf.columns)
        pvs_ind = columns.index('pvs')
        group_ind = columns.index('product_class')
        temp = self.pdf.dropna(axis=0).to_numpy()

        for idx in range(len(temp)):
            group = temp[idx][group_ind]
            if '|' in group:
                for sub in group.split('|'):
                    if sub in self.qclass_dict.keys():
                        self.qclass_dict[sub] = self.qclass_dict[sub].add_(temp[idx][pvs_ind])
                    else:
                        self.qclass_dict[sub] = temp[idx][pvs_ind]
            else:
                if group in self.qclass_dict.keys():
                    self.qclass_dict[group] = self.qclass_dict[group].add_(temp[idx][pvs_ind])
                else:
                    self.qclass_dict[group] = temp[idx][pvs_ind]
        
        for key in self.qclass_dict.keys():
            self.qclass_dict[key] = torchhd.soft_quantize(self.qclass_dict[key])
    
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

    def validate(self)->None:   
        skf = LeaveOneGroupOut()
        c = 0
        splits = skf.get_n_splits(groups=self.data['query_id'])

        query_vectors = self.data.groupby('query')['query'].apply(lambda q: sen2v(q.iloc[0], self.w2v, self.n_dimensions, self.device))
        self.data['query_vector'] = self.data['query'].map(query_vectors)
        acl,mrl,ndcgl,ltl = [],[],[],[]
        with torch.no_grad():
            pbar = tqdm(skf.split(self.data[['query','query_class','pvs']],self.data['label'],self.data['query_id']), total=splits, desc=f"LOGOCV| FOLD {c+1}/{splits}")
            for train_index, val_index in pbar:
                self.label = torchhd.random(num_vectors=3,dimensions=self.n_dimensions,device=self.device)
                y_true_test,y_pred_test=[],[]
                
                train_data= self.data.iloc[train_index]
                train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
                s_tr = len(train_data)

                s=0
                train_data_np = train_data.to_numpy()
                for row_t in train_data_np:
                    pbar.set_description(f'LOGOCV| FOLD {c+1} Training {s}/{s_tr}')
                    s+=1
                    label = row_t[5]
                    query = torch.stack([
                            row_t[6],
                            self.qclass_dict[row_t[3]] if row_t[3] in self.qclass_dict.keys() else torch.zeros(self.n_dimensions,device=self.device),
                            row_t[4]
                        ]).mul(self.query.weight)
                    
                    query = torchhd.soft_quantize(torchhd.multibundle(query))
                    scores = torchhd.cosine_similarity(query, self.label)
                    pred_ind = int(torch.argmax(scores))

                    if label != pred_ind:
                        self.label[pred_ind] = self.label[pred_ind].sub(query )
                        if scores[label] <= 0.95:
                            self.label[label] = self.label[label].add(query )
                    elif scores[label] <= 0 :
                        self.label[label] = self.label[label].add(query )
          
                
                val_data =self.data.iloc[val_index]
                val_data_np = val_data.to_numpy()
        
                s_te = len(val_data)
                timings_te=np.zeros((s_te,1))
           
                for row in range(s_te):
                    pbar.set_description(f'LOGOCV| FOLD {c+1} Training {s}/{s_tr} | Test {row}/{s_te}')  
                    starter = time.process_time()
                    label = val_data_np[row][5]
                    
                    query = torch.stack([
                            sen2v(val_data_np[0][2], self.w2v,self.n_dimensions,self.device),
                            self.qclass_dict[val_data_np[0][3]] if val_data_np[0][3] in self.qclass_dict.keys() else torch.zeros(self.n_dimensions,device=self.device),
                            val_data_np[row][4]
                        ]).mul(self.query.weight)
                    
                    query = torchhd.soft_quantize(torchhd.multibundle(query))
                    scores = torchhd.cosine_similarity(query, self.label)
                    pred_ind = int(torch.argmax(scores))

                    ender = time.process_time()
                    
                    timings_te[row] = (ender - starter)* 1000

                    y_true_test.append(label)
                    y_pred_test.append(pred_ind)

                        
                accuracy = accuracy_score(y_true_test, y_pred_test)
                laten = np.sum(timings_te)/len(timings_te)
                tmp = pd.DataFrame({'true': [SCORE[i] for i in y_true_test],'pred': [SCORE[i] for i in y_pred_test],'conf':y_pred_test}).sort_values(by=['pred','conf'], ascending=False)
                dcg = dcg_at_k(tmp['true'].values, 10)
                idcg = dcg_at_k(sorted(tmp['true'], reverse=True), 10)
                ndcg_n50 =  dcg / idcg if idcg > 0 else 0
                reciprocal_ranks = []
                combined_values = list(enumerate(zip(tmp.pred.values, tmp.conf.values)))
                sorted_values = sorted(combined_values, key=lambda x: (x[1][0], x[1][1]), reverse=True)
                sorted_indices = [i[0] for i in sorted_values]
                for rank, idx in enumerate(sorted_indices, start=1):
                        if tmp['true'].values[idx] > 0:
                            reciprocal_ranks.append(1 / rank)
                            break
                        else:
                            reciprocal_ranks.append(0)
                
                mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

                acl.append(accuracy)
                ndcgl.append(ndcg_n50)
                mrl.append(mrr)
                ltl.append(laten)

                pbar.set_postfix(train_accuracy=f"{np.sum(acl)/len(acl):.4f}",
                                latency_ms=f"{np.sum(ltl)/len(ltl):.4f}",
                                ndcg_10 = f'{np.sum(ndcgl)/len(ndcgl):.2f}',
                                mrr = f'{np.sum(mrl)/len(mrl):.2f}')
                c+=1
    def fit(self)->None:
        self.label = torchhd.random(num_vectors=3,dimensions=self.n_dimensions,device=self.device)
        with torch.no_grad():
            query_vectors = self.data.groupby('query')['query'].apply(lambda q: sen2v(q.iloc[0], self.w2v, self.n_dimensions, self.device))
            self.data['query_vector'] = self.data['query'].map(query_vectors)
            tmp = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
            pbar = tqdm(tmp[['query_vector','query_class','pvs','label']].to_numpy(), total=len(tmp), desc="Fitting WANDS")
            for row in pbar:
                label = row[3]
                query = torch.stack([
                    row[0],
                    self.qclass_dict[row[1]] if row[1] in self.qclass_dict.keys() else torch.zeros(self.n_dimensions,device=self.device),
                    row[2]
                ]).mul(self.query.weight)
    
                query = torchhd.soft_quantize(torchhd.multibundle(query))
                scores = torchhd.cosine_similarity(query, self.label)
                pred_ind = int(torch.argmax(scores))

                if label != pred_ind:
                    self.label[pred_ind] = self.label[pred_ind].sub(query )
                    if scores[label] <= 0.95:
                        self.label[label] = self.label[label].add(query )
                elif scores[label] <= 0 :
                    self.label[label] = self.label[label].add(query)

    def retrieve(self,retrieve, qclass,mx)->None:
        starter = time.process_time()
        self.exl,self.clas,self.il,self.score = [],[],[],[]
        cond =  qclass in self.qclass_dict.keys()
        if cond:
            filtered_pdf = self.pdf[self.pdf['product_class'].str.contains(qclass, na=False)]
            pbar = tqdm(filtered_pdf[['product_name', 'product_class', 'pvs']].to_numpy(), 
                        total=len(filtered_pdf), 
                        desc="Predicting WANDS")
        else:
            pbar = tqdm(self.pdf[['product_name','product_class','pvs']].to_numpy(), total=len(self.pdf), desc="Predicting WANDS")
        
        hvq = sen2v(retrieve, self.w2v,self.n_dimensions,self.device).mul(self.query.weight[0])
        qc = self.qclass_dict[qclass] if qclass in self.qclass_dict.keys() else torch.zeros(self.n_dimensions,device=self.device)
        hvqc = qc.mul(self.query.weight[1])
        for row in pbar:
            query = torch.stack([
                hvq,
                hvqc,
                row[2].mul(self.query.weight[2])
            ])
            query = torchhd.soft_quantize(torchhd.multibundle(query))
            scores = torchhd.cosine_similarity(query, self.label)
            pred_ind = int(torch.argmax(scores))
            self.exl.append(row[0])
            self.clas.append(pred_ind)
            self.score.append(scores[pred_ind])
            self.il.append(row[1])
    

        
        tmpp = pl.DataFrame({
            'item': self.exl,
            'group': self.il,
            'type': self.clas,
            'score': self.score
        })
        filtered_tmpp = tmpp.filter(pl.col('type') == 2)
        filtered_tmpp = filtered_tmpp.sort('type', 'score', descending=[True, True]) 

        ender = time.process_time()
        ms = (ender - starter)* 1000
        print(f" Time taken for this query: {ms} ms \n {filtered_tmpp.head(mx)}")
       