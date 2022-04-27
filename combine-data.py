import os
from os import listdir
from os.path import isfile, join
import glob
import pandas as pd

prog_file = 'combine-data.py'
path = os.path.dirname(os.path.abspath(prog_file))
print(path)

data_path='\data\cleaned_data'
complete_path = path + data_path
print(complete_path)

onlyfiles = [f for f in listdir(complete_path) if isfile(join(complete_path, f))]
print(onlyfiles)

path = 'data/cleaned_data/'
onlyfiles_path = [path + file_path for file_path in onlyfiles]

df = pd.concat(map(pd.read_csv, onlyfiles_path), ignore_index=True)
print(len(df))
df = df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
# print(df.head(10))
print(df.columns)

file_loc2 = 'data/combined_cleaned_data/combined.csv'
os.makedirs('data/combined_cleaned_data', exist_ok=True)  
df.to_csv(file_loc2) 