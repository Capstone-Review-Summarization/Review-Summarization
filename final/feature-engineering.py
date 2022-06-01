
import pandas as pd

df = pd.read_csv('data/combined_cleaned_data2/combined.csv')

# Creating a new column sentiment based on overall ratings
def sentiments(df):
  if df['overall'] > 3.0:
    return 'Positive'
  elif df['overall'] <= 3.0:
    return 'Negative'
df['sentiment'] = df.apply(sentiments, axis=1)

df.to_csv('data/combined_cleaned_data2/combined_with_fe.csv') 