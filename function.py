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



def lemmatize_sentence(sentence)->str:
    lemmatizer = WordNetLemmatizer()
    pos_tagged = pos_tag(sentence.split())
    lemmatized_sentence = []
    for word, tag in pos_tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return ' '.join(lemmatized_sentence)


def dcg_at_k(r, k)->float:
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.





        
