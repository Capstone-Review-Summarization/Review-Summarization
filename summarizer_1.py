from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import word_tokenize
import nltk

# Need to install additional package
# pip install sentencepiece

summarizer_tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_bbc")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_bbc")

def summarizer(content_list):
    summaries = []
    for i in range(len(content_list)):
        body = content_list[i]
        input_ids = summarizer_tokenizer(body, return_tensors="pt").input_ids
        output_ids = summarizer_model.generate(input_ids)[0]
        bert_summary = summarizer_tokenizer.decode(output_ids, skip_special_tokens=True)
        summaries.append(bert_summary)
    return summaries