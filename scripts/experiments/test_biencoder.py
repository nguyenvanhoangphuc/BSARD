import os
import json
from os.path import abspath, join

import torch
import pandas as pd

from utils.data import BSARDataset
from utils.eval import BiEncoderEvaluator
from models.trainable_dense_models import BiEncoder



if __name__ == '__main__':
    # 1. Load an already-trained BiEncoder.
    # "../../../output/training/Jul22-09-48-37/9"
    # "../../../output/training/Jul22-13-59-26/9"
    # /output/training/Jul23-15-46-59/9
    # /output/training/Jul23-18-14-22/2
    checkpoint_path = abspath(join(__file__, "../../../output/training/Jul23-18-14-22/2"))
    model = BiEncoder.load(checkpoint_path)

    # 2. Load the test set.
    test_queries_df = pd.read_csv(abspath(join(__file__, "../../../data/japanlaws/questions_ja_test.csv")))
    documents_df = pd.read_csv(abspath(join(__file__, "../../../data/japanlaws/articles_ja.csv")))
    test_dataset = BSARDataset(test_queries_df, documents_df)

    # 3. Initialize the Evaluator.
    evaluator = BiEncoderEvaluator(queries=test_dataset.queries, 
                                   documents=test_dataset.documents, 
                                   relevant_pairs=test_dataset.one_to_many_pairs, 
                                   score_fn=model.score_fn)

    # 4. Run trained model and compute scores.
    scores = evaluator(model=model,
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       batch_size=512)  

    # 5. Save results.
    output_path = abspath(join(__file__, "../../../output/zeroshotja/test-run"))
    os.makedirs(output_path, exist_ok=True)
    with open(join(output_path, 'test_scores_TwoTower_2.json'), 'w') as fOut:
        json.dump(scores, fOut, indent=2)
