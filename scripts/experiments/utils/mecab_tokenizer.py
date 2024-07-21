import MeCab
wakati = MeCab.Tagger("-Owakati")
# wakati.parse("pythonが大好きです").split()
# tagger = MeCab.Tagger()
# print(tagger.parse("pythonが大好きです"))

def tokenize_mecab(text):
    return wakati.parse(text).split()

print(tokenize_mecab("pythonが大好きです"))