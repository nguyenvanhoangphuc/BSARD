import csv
import pandas as pd

df_laws = pd.read_csv('articles_ja.csv', encoding='utf-8')

law_id = "労働基準法/第三章　賃金"
article_id = "1"

# ids = df_laws[df_laws['law_id'] == law_id]['article_id'].values
ids = []
for i, row in df_laws.iterrows():
    if row['law_id'] == law_id and str(row['article_id']) == article_id:
        print(row['id'])
        ids.append(row['id'])
print(ids)
print(len(ids))