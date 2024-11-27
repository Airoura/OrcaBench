import os
import json
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from nltk.util import ngrams
from sacrebleu.metrics import BLEU

def check_dirs(directory):
   if not os.path.exists(directory):
       os.makedirs(directory)

def save_dic2json(file_path, dic):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

def singleQuoteToDoubleQuote(singleQuoted):

    '''

    convert a single quoted string to a double quoted one

    Args:

        singleQuoted(string): a single quoted string e.g. {'cities': [{'name': "Upper Hell's Gate"}]}

    Returns:

        string: the double quoted version of the string e.g. 

    see

       - https://stackoverflow.com/questions/55600788/python-replace-single-quotes-with-double-quotes-but-leave-ones-within-double-q 

    '''

    cList=list(singleQuoted)

    inDouble=False;

    inSingle=False;

    for i,c in enumerate(cList):

        #print ("%d:%s %r %r" %(i,c,inSingle,inDouble))

        if c=="'":

            if not inDouble:

                inSingle=not inSingle

                cList[i]='"'

        elif c=='"':

            inDouble=not inDouble

    doubleQuoted="".join(cList)    

    return doubleQuoted

def get_elements_between_braces(s):
        start = s.find('{')
        end = s.rfind('}')
        if start == -1 or end == -1:
           return None
        return s[start:end + 1]

def load_json(r):
    result = get_elements_between_braces(r)
    json_str_replaced = singleQuoteToDoubleQuote(result)
    return json.loads(json_str_replaced)

def load_json_or_pure(r):
    res = {
        "Tweet Content": r
    }
    try:
        result = get_elements_between_braces(r)
        json_str_replaced = singleQuoteToDoubleQuote(result)
        res = json.loads(json_str_replaced)
    except Exception as e:
        # print(e)
        pass
    return res

# def calculate_bleu(reference, candidate):
#     bleu_score = sentence_bleu([reference], candidate)
#     return bleu_score
    
def calculate_bleu(references, pred):
    bleu_scorer = BLEU(effective_order=True)
    bleu = bleu_scorer.sentence_score(pred, references).score
    return bleu

def calculate_distinct(text):
    
    tokens = text.split()
    
    distinct_1gram = len(set(tokens))
    
    distinct_2gram = len(set(ngrams(tokens, 2)))
    
    distinct_3gram = len(set(ngrams(tokens, 3)))
    
    distinct = (distinct_1gram + distinct_2gram + distinct_3gram) / len(tokens)
    return distinct

def cosine_similarity(vec1, vec2):
    
    dot_product = np.dot(vec1, vec2)
    
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

def euclidean_distance(vec1, vec2):
    
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1, vec2):
    
    return np.sum(np.abs(vec1 - vec2))
