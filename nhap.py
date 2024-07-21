# mở file requirements.txt và thay bằng nội dung mới sau đó lưu lại file đó


# # Đọc file requirements.txt và in ra màn hình
with open("requirements.txt", "r") as f:
    print(f.read())


content = """numpy
pandas
torch
transformers
sentence-transformers
tensorboard
gensim
spacy
seaborn
dash
wordcloud
datasets
fasttext-wheel
scipy==1.10.1"""
with open("requirements.txt", "w") as f:
    f.write(content)
# Đọc file requirements.txt và in ra màn hình
with open("requirements.txt", "r") as f:
    print(f.read())
