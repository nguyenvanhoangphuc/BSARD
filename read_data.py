import pandas as pd
import numpy as np

# # read file articles.csv
# df_articles = pd.read_csv("/data/bsard_v1/articles_fr.csv")
# # print(df_articles.head())
# df_articles.info()

# print("df_articles.iloc[0]['article']", df_articles.iloc[0]['article'])


# read file questions_fr_test.csv
dfQ_test = pd.read_csv("/data/bsard_v1/questions_fr_test.csv")
# print(dfQ_test.head())
dfQ_test.info()

print("dfQ_test.iloc[0]['id']", dfQ_test.iloc[0]['id'])
print("dfQ_test.iloc[0]['category']", dfQ_test.iloc[0]['category'])
print("dfQ_test.iloc[0]['subcategory']", dfQ_test.iloc[0]['subcategory'])
print("dfQ_test.iloc[0]['question']", dfQ_test.iloc[0]['question'])
print("dfQ_test.iloc[0]['extra_description']", dfQ_test.iloc[0]['extra_description'])
print("dfQ_test.iloc[0]['article_ids']", dfQ_test.iloc[0]['article_ids'])