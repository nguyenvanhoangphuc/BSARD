# from datasets import load_dataset

# repo = "maastrichtlawtech/bsard"

# # Load corpus of statutory articles.
# articles = load_dataset(repo, name="corpus", split="corpus")

# # Load training questions.
# train_questions = load_dataset(repo, name="questions", split="train")
# # train_negatives = load_dataset(repo, name="negatives", split="train")

# # Optional: load synthetic questions for extra training samples.
# synthetic_questions = load_dataset(repo, name="questions", split="synthetic")
# # synthetic_negatives = load_dataset(repo, name="negatives", split="synthetic")

# # Load testing questions.
# test_questions = load_dataset(repo, name="questions", split="test")

# # Lưu articles vào file articles_fr.csv
# articles.to_csv("/data/bsard_v1/articles_fr.csv", index=False)  

# # Lưu train_questions vào file questions_fr_train.csv
# train_questions.to_csv("/data/bsard_v1/questions_fr_train.csv", index=False)

# # Lưu synthetic_questions vào file questions_fr_synthetic.csv
# synthetic_questions.to_csv("/data/bsard_v1/questions_fr_synthetic.csv", index=False)

# # Lưu test_questions vào file questions_fr_test.csv
# test_questions.to_csv("/data/bsard_v1/questions_fr_test.csv", index=False)

from datasets import load_dataset
import pandas as pd

repo = "maastrichtlawtech/bsard"

# Load corpus of statutory articles.
articles = load_dataset(repo, name="corpus")

# Load training questions.
train_questions = load_dataset(repo, name="questions", split="train")

# Optional: load synthetic questions for extra training samples.
synthetic_questions = load_dataset(repo, name="questions", split="synthetic")

# Load testing questions.
test_questions = load_dataset(repo, name="questions", split="test")

# Lưu articles vào file articles_fr.csv
df_articles = pd.DataFrame(articles['corpus'])  # Giả sử 'articles' chỉ có một split
df_articles.to_csv("/data/bsard_v1/articles_fr.csv", index=False)

# Lưu train_questions vào file questions_fr_train.csv
df_train_questions = pd.DataFrame(train_questions)
df_train_questions.to_csv("/data/bsard_v1/questions_fr_train.csv", index=False)

# Lưu synthetic_questions vào file questions_fr_synthetic.csv
df_synthetic_questions = pd.DataFrame(synthetic_questions)
df_synthetic_questions.to_csv("/data/bsard_v1/questions_fr_synthetic.csv", index=False)

# Lưu test_questions vào file questions_fr_test.csv
df_test_questions = pd.DataFrame(test_questions)
df_test_questions.to_csv("/data/bsard_v1/questions_fr_test.csv", index=False)
