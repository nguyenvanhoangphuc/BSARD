# đọc file json chứa dữ liệu văn bản luật tiếng việt
import json
import pandas as pd

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

if __name__ == '__main__':
    file_path = 'legal_corpus.json'
    data = read_json_file(file_path)
    print(len(data))

    # create new dataframe 
    new_data = []
    count_id = 1
    for item in data:
        for art in item['articles']:
            new_data.append({
                'id': count_id, 
                'law_id': item['law_id'], 
                'article_id': str(art['article_id']),
                'article_title': art['title'],
                'article_text': art['text'],
                'article': item['law_id'] + art['title'] + art['text']
            })
            count_id += 1

    print(len(new_data))
    df_new = pd.DataFrame(new_data)
    print(df_new.head())
    df_new.to_csv('articles_ja.csv', index=False)