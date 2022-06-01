import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
import numpy as np
from rouge import Rouge
import random
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from summarizer import Summarizer

# may need to download some of the nltk stuff and the bert-extractive-summarizer, uncomment the below lines to download it
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# pip install pytorch-transformers
# pip install bert-extractive-summarizer

# Get the input from the webscraper in a list of strings format
# Change input method for dynamic content
reviews_scraped = ['  I would not recommend this product for winters.', '  Good', '  This pair of gloves is not for very cold temperatures, but is of high quality build and should be able to handle most East Coast urban winter environments and outdoor activities.  If you need to use it for 20 F or lower temperatures with high wind chill factor, then you either need to use it with a pair of glove-sleeves/inner gloves or get a pair of heavier gloves. The stitching and seams are well executed.  The profile is slim and they do not feel bulky on my hands.  I can wear them with business and causal wears.  Index fingers on both gloves can operate smart phones and tablets.  Also, there is a micro fastex style buckle that can clip both gloves together. I highly recommend this pair of gloves.', "  They look good, and great quality. The touch screen tips work but not well. They do not keep your hands and fingers warm at very cold weather! I will have to find another gloves for cold weather. For example - Columbia Men's Northport Insulated Softshell Glove.I will add a little.I am not happy with the condition of my hands (fingers) in cold weather with these gloves. At 0 degrees Celsius, you can still just walk in them. But as soon as the temperature drops to at least -5 degrees Celsius, your fingers just freeze. In order for these gloves to be comfortable, you need to constantly actively do something. Otherwise, it will simply not work to warm your hands (fingers).It is a pity that these gloves cannot warm my hands (fingers) at temperatures below 0 degrees Celsius during normal walking use.", "  I normally wear size L and this size L fits very well. There is enough room on the palm and back of hands so that it doesn't restrict movement. It is comfortable when holding trekking poles or driving wheel without too much pressure on knuckle area. The fit on fingers are perfect for me, except that the thumbs are 1cm too long. My thumbs can only reach the end of gloves when I do a thumb up gesture. I don't think my activity involves the thumb up gesture at all. A walk outside at 4C temperature initially feels a bit cold on hands. After a while the blood circulation is more active my hands feel warm. Overall I like the gloves very much.", '  I took the advice of all the reviews and ordered a size up from normal. Big mistake, these gloves for me like any other gloves this style i have bought. The gloves i ordered in large are baggy on my finger tips and i should have went with my normal medium size order. The gloves themselves are awesome and are great quality. The touch screen tips work as they say.', '  They look good, fit snug and the touch screen fingers actually work. They seem well put together. They do not keep your hands and fingers warm once about 40deg out or lower! I will have to find another option for those cooler nights.']

# Converting data from a list to a dataframe to preserve the original data
reviews_scraped_df = pd.DataFrame(reviews_scraped, columns = ['reviewText'])

# Adding empty column in the dataframe to store the cleaned text
reviews_scraped_df['reviewCleaned'] = ''

# Preprocessing Steps

# Getting the PoS of the string text
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # stripping the text from any leading or trailing whitespaces
    text.strip()
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove duplicates
    text = list(set(text))
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return text

for review in reviews_scraped:
    # Replace nextline with space for format
    reviews_scraped_df['reviewText'] = reviews_scraped_df['reviewText'].str.replace('\n', ' ')
    # clean text data
    reviews_scraped_df["reviewCleaned"] = reviews_scraped_df["reviewText"].apply(lambda x: clean_text(str(x)))

# Processing Data

# converting datatype from pandas series to str to help in vectorization
reviews_scraped_df.reviewCleaned = reviews_scraped_df.reviewCleaned.astype(str)
# storing all the cleaned data in a list
corpus = reviews_scraped_df['reviewCleaned'].tolist()

# vectoriztion of input data to feed the model
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(reviews_scraped_df['reviewCleaned'].values)
X = tokenizer.texts_to_sequences(reviews_scraped_df['reviewCleaned'].values)
X = pad_sequences(X)

# Loading the model
model = tf.keras.models.load_model('model2.hdf5')

# Sentiment Analysis

positive_reviews = pd.DataFrame(columns = ['reviewText', 'reviewCleaned'])
negative_reviews = pd.DataFrame(columns = ['reviewText', 'reviewCleaned'])

# Splitting the data in positive and negative reviews using sentiment analysis
# for i in range(len(reviews_scraped_df['reviewCleaned'])):
#     review = [reviews_scraped_df['reviewCleaned'][i]]
#     #vectorizing the review by the pre-fitted tokenizer instance
#     review = tokenizer.texts_to_sequences(review)
#     #padding the review to have exactly the same shape as `embedding_2` input
#     review = pad_sequences(review, maxlen=661, dtype='int32', value=0)
#     # print(review)
#     sentiment = model.predict(review,batch_size=1,verbose = 2)[0]
#     if(np.argmax(sentiment) == 1):
#         df_temp = pd.DataFrame({'reviewText': [reviews_scraped_df['reviewText'][i]], 'reviewCleaned': [reviews_scraped_df['reviewCleaned'][i]]})
#         positive_reviews = pd.concat([positive_reviews, df_temp], ignore_index = True, axis = 0)
#     elif (np.argmax(sentiment) == 0):
#         df_temp = pd.DataFrame({'reviewText': [reviews_scraped_df['reviewText'][i]], 'reviewCleaned': [reviews_scraped_df['reviewCleaned'][i]]})
#         negative_reviews = pd.concat([negative_reviews, df_temp], ignore_index = True, axis = 0)

# Clustering, Weak Reference and Text Summarization

# corpus_for_positive_clustering = positive_reviews['reviewText'].tolist()
# corpus_for_negative_clustering = negative_reviews['reviewText'].tolist()

rouge = Rouge()

def clustering(corpus_for_clustering):
    cluster = []
    clusters = []
    token_len = 0
    rouge = Rouge()
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
    
    for i in range(len(clusters)):
        tokenized_text = word_tokenize(str(clusters[i]).strip("[]").replace("'", ""))
        cluster_len = len(tokenized_text)
        if (cluster_len > 512):
            new_token_cluster = tokenized_text[0:128] + tokenized_text[-382:]
            cluster_text = TreebankWordDetokenizer().detokenize(tokenized_text)
            clusters = clusters[:i]+[cluster_text]+clusters[i+1:]

    return clusters

# def weak_ref_ext(content_list):
#     total_score = 0
#     review_f1score_pair = {}
#     wre = []
#     for cluster in content_list:
#         review_list = cluster
#         review_list = list(set(review_list))
#         for review in review_list:
#             if(review !='\n'):
#                 for other_review in review_list:
#                     if(other_review != review):
#                         score = rouge.get_scores(review, other_review)
#                         total_score = total_score + score[0].get('rouge-1').get('f')/(len(review_list) - 1)
#                 review_f1score_pair[review] = total_score
#                 total_score = 0
#         {k: v for k, v in sorted(review_f1score_pair.items(), key=lambda item: item[1])}
#         wre.append(list(review_f1score_pair.keys())[-1])
#         review_f1score_pair = {}
#     return wre

def summarizer(content_list):
    summaries = []
    bert_model = Summarizer()
    for i in range(len(content_list)):
        body = ' '.join([str(elem) for elem in content_list[i]])
        bert_summary = ''.join(bert_model(body, min_length=20))
        summaries.append(bert_summary)
    return summaries

def final_steps(content_list):
    while (len(content_list) != 1):
        text_clusters = clustering(content_list)
        # for i in range(len(text_clusters[0])):
        #     text_clusters[0][i].strip()
        # wre_clusters = weak_ref_ext(text_clusters)
        # wre_text_clusters = clustering(wre_clusters)
        final_summary = summarizer(text_clusters)
        content_list = final_summary
    return content_list

positive_summary = final_steps(reviews_scraped_df['reviewText'].tolist())
# negative_summary = final_steps(negative_reviews['reviewText'].tolist())
print(positive_summary)
# print(negative_summary)