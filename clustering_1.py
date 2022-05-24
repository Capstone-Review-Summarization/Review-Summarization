from nltk.tokenize import word_tokenize
import random
from rouge import Rouge
import pandas as pd
from nltk.tokenize.treebank import TreebankWordDetokenizer

rouge = Rouge()

def clustering(corpus_for_clustering):
    cluster = []
    clusters = []
    token_len = 0

    if(len(corpus_for_clustering) <= 2):
        clusters = [review for review in corpus_for_clustering]

    while len(corpus_for_clustering) > 2:
        pivot_data = random.choice(corpus_for_clustering)
        token_len = len(word_tokenize(pivot_data))
        cluster = [pivot_data]
        df_cluster = pd.DataFrame(columns = ['text', 'rouge-1 score'])
        for j in range(len(corpus_for_clustering)):        # getting rouge-1 f1 score for all data wrt to pivot_data
            if (pivot_data != corpus_for_clustering[j]):
                scores = rouge.get_scores(pivot_data, corpus_for_clustering[j])
                df_cluster = df_cluster.append({'text': corpus_for_clustering[j], 'rouge-1 score': scores[0].get('rouge-1').get('f')}, ignore_index=True)
        df_cluster.sort_values("rouge-1 score", axis = 0, ascending = False, inplace = True, na_position ='last')
        token_len = token_len + len(word_tokenize(df_cluster['text'][0])) + len(word_tokenize(df_cluster['text'][1]))
        already_in_cluster = [pivot_data, df_cluster['text'][0], df_cluster['text'][1]]
        cluster.append(df_cluster['text'][0])
        cluster.append(df_cluster['text'][1])
        for k in range(2, len(df_cluster)):
            if(len(word_tokenize(df_cluster['text'][k])) + token_len < 512):
                token_len = token_len + len(word_tokenize(df_cluster['text'][k]))
                cluster.append(df_cluster['text'][k])
                df_cluster.drop(k, inplace=True)
                already_in_cluster.append(df_cluster['text'][k])
            else:
                break
        corpus_for_clustering = [ review for review in corpus_for_clustering if review not in already_in_cluster ]
        del df_cluster
        cluster = []
    
    for i in range(len(clusters)):
        tokenized_text = word_tokenize(str(clusters[i]).strip("[]").replace("'", ""))
        cluster_len = len(tokenized_text)
        if (cluster_len > 512):
            new_token_cluster = tokenized_text[0:128] + tokenized_text[-382:]
            # print(len(new_token_cluster))
            # print(tokenized_text)
            # print(new_token_cluster)
            # more = more + 1
            cluster_text = TreebankWordDetokenizer().detokenize(tokenized_text)
            clusters = clusters[:i]+[cluster_text]+clusters[i+1:]
    
    return clusters


