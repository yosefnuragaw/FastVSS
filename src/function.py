import torch
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from gensim.models import Word2Vec
import re
import numpy as np
from datetime import datetime    

def clean_string(text)->str:
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = text.strip()
        return text
    else:
        return ''
    
def status(text:str,verbose:bool)->None:
    if verbose:
        print(f"{datetime.now()}|STATUS:{text}")

def sen2v(
        text:str,
        model:Word2Vec,
        n_dimension:int=128,
        device=torch.device
        )->torch.Tensor:
    lemma = lemmatize_sentence(clean_string(text)).split(' ')
    temp = np.zeros(n_dimension)
    count = 0
    for token in lemma:
        if token in model.wv:
            count+=1
            temp = np.add(temp,model.wv[token])

    if count == 0:
        return torch.from_numpy(temp).to(device)
    return torch.from_numpy(temp / count).to(device)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_sentence(sentence)->str:
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(sentence.split())
    lemmatized_sentence = []
    for word, tag in pos_tagged:
        wordnet_pos = get_wordnet_pos(tag) 
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_sentence)


def dcg_at_k(arr, k)->float:
    arr = np.asfarray(arr)[:k]
    if arr.size:
        return np.sum(arr / np.log2(np.arange(2, arr.size + 2)))
    return 0.

def ndcg(arr,k)->float:
    idcg = dcg_at_k(sorted(arr, reverse=True), k)
    return dcg_at_k(arr, k) / idcg if idcg > 0 else 0

def mrr(arr)->float:
    reciprocal_ranks = []
    combined_values = list(enumerate(zip(arr.pred.values, arr.conf.values)))
    sorted_values = sorted(combined_values, key=lambda x: (x[1][0], x[1][1]), reverse=True)
    sorted_indices = [i[0] for i in sorted_values]
    for rank, idx in enumerate(sorted_indices, start=1):
            if arr['true'].values[idx] > 0:
                reciprocal_ranks.append(1 / rank)
                break
            else:
                reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)




        
