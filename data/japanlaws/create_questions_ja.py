import json
import csv
import pandas as pd

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

if __name__ == '__main__':
    data_full = read_json_file('train_12x7_retrieval.json')  # validation_12x7_retrieval # train_12x7_retrieval
    data = data_full['items']

    # read file csv articles_ja.csv
    df_laws = pd.read_csv('articles_ja.csv', encoding='utf-8')

    questions = []
    count_id = 1
    for item in data:
        article_ids = []
        for article in item['relevant_articles']: 
            law_id = article['law_id']
            article_id = article['article_id']
            # print(law_id, article_id)
            # ids = df_laws[(df_laws['law_id'] == law_id) & (df_laws['article_id'] == article_id)]['id'].values
            ids = []
            for i, row in df_laws.iterrows():
                if row['law_id'] == law_id and str(row['article_id']) == article_id:
                    # print(row['id'])
                    ids.append(row['id'])
            # print(ids)

            if len(ids) > 0:
                article_ids.append(str(ids[0]))
            if len(ids) > 1: 
                print('Error: duplicate article')
        if (len(article_ids) == 0):
            continue
        question = {
            'id': count_id,
            'question_short': item['question_short'],
            'question': item['question'],
            'question_full': item['question_full'],
            'article_ids': ",".join(article_ids),
        }
        count_id += 1
        questions.append(question)
    
    print(len(questions))
    df_new = pd.DataFrame(questions)
    print(df_new.head())
    df_new.to_csv('questions_ja_train.csv', index=False)