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
    # output/training/Jul24-08-21-05/2 # 13
    # output/training/Jul2407/2 # 16
    # output/training/Jul24-08-37-45/19   #36 epochs
    # BSARD/output/training/Jul24-10-16-25/99 # 136e two-tower 512 512 xlm 16 (japan)   - best japan
    # BSARD/output/training/Aug14-09-42-18/99 # full marco two-tower
    # BSARD/output/training/Aug14-12-47-24/99 # full marco siamese
    # BSARD/output/training/Oct07-07-17-36/0 # siamese (full marco)
    # /BSARD/output/training/Oct07-10-30-02/33  # two-tower (full marco)
    # BSARD/output/training/Oct07-08-51-35/30 #  siamese (full marco)
    # BSARD/output/training/Oct07-08-51-35/89 # 90e siamese (full marco)
    # BSARD/output/training/Oct08-19-05-50/16 # 50e two-tower (full marco)
    # BSARD/output/training/Oct10-10-42-02/36 # siamese 512 512 (1110)
    # BSARD/output/training/Oct10-10-42-02/49 # siamese 512 512 batch 16 (1310)
    # BSARD/output/training/Oct14-10-15-10/37 # siamese 256 256 bert 16 (1410)
    # BSARD/output/training/Oct14-10-36-11/40 # two-tower 256 256 bert 16 (1410)
    # /Oct14-10-15-10/49 # siamese 256 256 bert 16 50e (1410)
    # /Oct14-10-36-11/49 # two-tower 256 256 bert 16 50e (1410)
    # BSARD/output/training/Oct16-02-24-07/49 # two-tower 512 512 bert 16 50e (1610) 
    # BSARD/output/training/Oct23-07-32-45/49 # sia 128 128 bert 16 50e (2310)  - best ms20k 
    # BSARD/output/training/Oct23-07-37-01/49 # two 128 128 bert 16 50e (2310)  - best ms20k 
    # BSARD/output/training/Oct24-06-46-58/49 # sia 512 512 xlm 4 50e (2410)    
    # BSARD/output/training/Oct28-08-37-28/14 # sia 512 512 xlm 4 100e (2810)   - best japan
    # BSARD/output/training/Oct28-08-09-26/19 # sia 512 512 xlm 8 20e (2810)
    # BSARD/output/training/Oct29-02-33-06/14 # 15e sia 512 512 xlm 8 20e (2910)
    # BSARD/output/training/Oct29-08-18-29/49 # 50e sia 512 512 bert 8 (2910)
    # BSARD/output/training/Oct29-08-20-14/49 # 50e two 512 512 bert 8 (2910)
    # BSARD/output/training/Oct29-09-34-55/49 # 100e sia 512 512 bert 8 (2910)  - best japan
    # BSARD/output/training/Oct29-09-33-04/49 # 100e two 512 512 bert 8 (2910)  - best japan
    checkpoint_path = abspath(join(__file__, "../../../output/training/Oct23-07-37-01/49"))   
    model = BiEncoder.load(checkpoint_path)

    # 2. Load the test set.
    # test_queries_df = pd.read_csv(abspath(join(__file__, "../../../data/japanlaws/questions_ja_test.csv")))
    # test_queries_df = pd.read_csv(abspath(join(__file__, "../../../data/BEIR/fiqa_BSARD/questions_fiqa_test.csv")))
    test_queries_df = pd.read_csv(abspath(join(__file__, "../../../data/20k_ms_marco_BSARD/questions_ms_marco_20k_test.csv")))
    # documents_df = pd.read_csv(abspath(join(__file__, "../../../data/japanlaws/articles_ja.csv")))
    # documents_df = pd.read_csv(abspath(join(__file__, "../../../data/BEIR/fiqa_BSARD/articles_fiqa.csv")))
    documents_df = pd.read_csv(abspath(join(__file__, "../../../data/20k_ms_marco_BSARD/articles_ms_marco_20k.csv")))
    test_dataset = BSARDataset(test_queries_df, documents_df)

    # 3. Initialize the Evaluator.
    evaluator = BiEncoderEvaluator(queries=test_dataset.queries, 
                                   documents=test_dataset.documents, 
                                   relevant_pairs=test_dataset.one_to_many_pairs, 
                                   score_fn=model.score_fn)

    # 4. Run trained model and compute scores.
    scores = evaluator(model=model,
                       device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                       batch_size=8)  

    # 5. Save results.
    # output_path = abspath(join(__file__, "../../../output/zeroshotBEIR"))
    # output_path = abspath(join(__file__, "../../../output/zeroshotja/test-run"))
    output_path = abspath(join(__file__, "../../../output/finetune_marco"))
    os.makedirs(output_path, exist_ok=True)
    with open(join(output_path, 'ms20k_two_bert_128_16_bs50.json'), 'w') as fOut:
    # with open(join(output_path, 'sia_one_shot_beir_fiqa.json'), 'w') as fOut:
        json.dump(scores, fOut, indent=2)
