from nltk.tokenize import word_tokenize
from rouge import Rouge
import random
import pandas as pd

def clustering(corpus_for_clustering):
    token_len = 0
    rouge = Rouge()
    cluster = []
    clusters = []
    already_in_cluster = []
    while len(corpus_for_clustering) != 0:
        pivot_data = random.choice(corpus_for_clustering)
        token_len = len(word_tokenize(pivot_data))
        cluster = [pivot_data]
        df_cluster = pd.DataFrame(columns = ['text', 'rouge-1 score'])
        for j in range(len(corpus_for_clustering)):        # getting rouge-1 f1 score for all data wrt to pivot_data
            if (corpus_for_clustering[j] != pivot_data):
                scores = rouge.get_scores(pivot_data, corpus_for_clustering[j])
                df_temp = pd.DataFrame({'text': [corpus_for_clustering[j]], 'rouge-1 score': [scores[0].get('rouge-1').get('f')]})
                df_cluster = pd.concat([df_cluster, df_temp], ignore_index = True, axis = 0)
        
        df_cluster.sort_values("rouge-1 score", axis = 0, ascending = False, inplace = True, na_position ='last')
        already_in_cluster = [pivot_data]
        for k in range(len(df_cluster)):
            if(len(word_tokenize(df_cluster['text'][k])) + token_len < 512):
                token_len = token_len + len(word_tokenize(df_cluster['text'][k]))
                cluster.append(df_cluster['text'][k])
                # df_cluster.drop(k, inplace=True)
                already_in_cluster.append(df_cluster['text'][k])
            else:
                break
        corpus_for_clustering = [ review for review in corpus_for_clustering if review not in already_in_cluster ]
        del df_cluster
        clusters.append(cluster)
        cluster = []
    return clusters

corpus = "input"

remove_corpus = []
for review in corpus:
    if (len(word_tokenize(review)) > 512):
        remove_corpus.append(review)
        more = more + 1

corpus = [ review for review in corpus if review not in remove_corpus ]
clusters = clustering(corpus)