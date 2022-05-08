import os
from os import listdir
from os.path import isfile, join
import time

prog_file = 'sample-data.py'
path = os.path.dirname(os.path.abspath(prog_file))

data_path='\data\org_data'
complete_path = path + data_path

# Listing out all the JSON files (dataset)
onlyfiles = [f for f in listdir(complete_path) if isfile(join(complete_path, f))]
onlyfiles = [e for e in onlyfiles if e not in ('Books_5.json', 'Clothing_Shoes_and_Jewelry_5.json', 'Electronics_5.json', )]
# print(onlyfiles)

data_file = 'Video_Games_5.json'

import pandas as pd

start_time = time.time()

file_loc1 = 'data/org_data/' + data_file
df = pd.read_json(file_loc1, lines=True)

if (len(df) <= 150000):
    df = df

else:
    df = df.head(150000)

file_loc2 = 'data/data_small/' + data_file
os.makedirs('data/data_small', exist_ok=True)  
df.to_csv(file_loc2) 

print(f"File: {data_file} --- %s seconds " % (time.time() - start_time))