# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
# Sử dụng các file dữ liệu sau:
- articles.csv dùng thuộc tính article, id (cx chính là số thứ tự, nên nó không cần truy cập vào thuộc tính này)
=> articles.csv có thể chỉ có một thuộc tính duy nhất là article và chỉ cần sắp xếp nó theo id từ 1 => end.
- questions_fr_test.csv dùng thuộc tính article_ids, question
=> file này dùng 2 cái quan trọng nhất là question => tìm ra các câu article liên quan trong tập corpus, xong rồi đánh giá so sánh với các article_ids trong thuộc tính article_ids.

python scripts/experiments/run_zeroshot_evaluation.py \
    --articles_path </path/to/articles.csv> \
    --test_questions_path </path/to/questions_test.csv> \
    --retriever_model {tfidf, bm25} \ 
    --lem \ 
    --output_dir </path/to/output>

python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/bsard_v1/articles_fr.csv --test_questions_path data/bsard_v1/questions_fr_test.csv --retriever tfidf --lem --output_dir /output/zeroshot/test-run
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/bsard_v1/articles_fr.csv --test_questions_path data/bsard_v1/questions_fr_test.csv --retriever bm25 --lem --output_dir /output/zeroshot/test-run
# japan
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/japanlaws/articles_ja.csv --test_questions_path data/japanlaws/questions_ja_test.csv --retriever tfidf --lem --output_dir /output/zeroshotja/test-run
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/japanlaws/articles_ja.csv --test_questions_path data/japanlaws/questions_ja_test.csv --retriever bm25 --lem --output_dir /output/zeroshotja/test-run

bash scripts/experiments/utils/download_embeddings.sh

python scripts/experiments/run_zeroshot_evaluation.py \
    --articles_path </path/to/articles.csv> \
    --test_questions_path </path/to/questions_test.csv> \
    --retriever {word2vec, fasttext, camembert} \ 
    --lem \ # [Only for word2vec and fastText] Lemmatize both articles and questions as pre-processing.
    --output_dir </path/to/output>

# run fasttext, word2vec no train
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/bsard_v1/articles_fr.csv --test_questions_path data/bsard_v1/questions_fr_test.csv --retriever word2vec --lem --output_dir output/zeroshot/test-run
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/bsard_v1/articles_fr.csv --test_questions_path data/bsard_v1/questions_fr_test.csv --retriever fasttext --lem --output_dir output/zeroshot/test-run
# run bert no train
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/bsard_v1/articles_fr.csv --test_questions_path data/bsard_v1/questions_fr_test.csv --retriever bert --output_dir output/zeroshot/test-run
# japan
python scripts/experiments/run_zeroshot_evaluation.py --articles_path data/japanlaws/articles_ja.csv --test_questions_path data/japanlaws/questions_ja_test.csv --retriever bert --output_dir /output/zeroshotja/test-run

python scripts/experiments/train_biencoder.py

python scripts/experiments/test_biencoder.py