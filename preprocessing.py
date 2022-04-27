import os
from os import listdir
from os.path import isfile, join
import time

prog_file = 'preprocessing.py'
path = os.path.dirname(os.path.abspath(prog_file))

data_path='\data\sample_data'
complete_path = path + data_path

# Listing out all the JSON files (dataset)
onlyfiles = [f for f in listdir(complete_path) if isfile(join(complete_path, f))]
# print(onlyfiles)

import nltk
from nltk.corpus import wordnet

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

import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # print(text)
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove duplicates
    text = list(set(text))
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
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

for data_file in onlyfiles:
    start_time = time.time()
    file_loc1 = 'data/sample_data/' + data_file
    df = pd.read_csv(file_loc1)
    # Replace nextline with space for format
    df['reviewText'] = df['reviewText'].str.replace('\n', ' ')
    # clean text data
    df["review_clean"] = df["reviewText"].apply(lambda x: clean_text(str(x)))
    # print(df["review_clean"])

    file_loc2 = 'data/cleaned_data/' + data_file
    os.makedirs('data/cleaned_data', exist_ok=True)  
    df.to_csv(file_loc2)  
    print("--- %s seconds ---" % (time.time() - start_time))