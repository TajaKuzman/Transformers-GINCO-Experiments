import os
import json
import parse
import fasttext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

train_labels = ['__label__Legal/Regulation', '__label__Opinionated_News', '__label__News/Reporting', '__label__Forum', '__label__Correspondence', '__label__Invitation', '__label__Instruction', '__label__Recipe', '__label__Opinion/Argumentation', '__label__Promotion_of_Services', '__label__Promotion', '__label__List_of_Summaries/Excerpts', '__label__Promotion_of_a_Product', '__label__Call', '__label__Review', '__label__Other', '__label__Information/Explanation', '__label__Interview', '__label__Prose', '__label__Research_Article', '__label__Announcement']

STR_TO_NUM = {s: i for i, s in enumerate(train_labels)}
NUM_TO_STR = {i: s for i, s in enumerate(train_labels)}


def parse_fasttext_file(path: str, encode=True):
    """Reads fasttext formatted file and returns dataframe."""
    with open(path, "r") as f:
        content = f.readlines()
    pattern = "{label} {text}\n"
    p = parse.compile(pattern)

    labels, texts = list(), list()
    for line in content:
        rez = p.parse(line)
        if rez is not None:
            if rez["label"] == '__label__Promotion_of_services':
                labels.append('__label__Promotion_of_Services')
            elif rez["label"] == '__label__Promotion_of_a_product':
                labels.append('__label__Promotion_of_a_Product')
            else:
                labels.append(rez["label"])
            texts.append(rez["text"])
        else:
            pass
            #print("error parsing line ", line)
    if encode:
        labels = [STR_TO_NUM[i] for i in labels]
    return pd.DataFrame(data={"text": texts, "labels": labels})



def train_model(train_df, NUM_EPOCHS=30):
    from simpletransformers.classification import ClassificationModel
    model_args = {
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": 1e-5,
        "overwrite_output_dir": True,
        "train_batch_size": 32,
        "no_save": True,
        "no_cache": True,
        "overwrite_output_dir": True,
        "save_steps": -1,
        "max_seq_length": 512,
        "silent": True
    }

    model = ClassificationModel(
        "camembert", "EMBEDDIA/sloberta",
        num_labels = 21,
        use_cuda = True,
        args = model_args
    )
    model.train_model(train_df)
    return model

def eval_model(test_df, model):
    y_true_enc = test_df.labels
    y_pred_enc = model.predict(test_df.text.tolist())[0]

    y_true = [NUM_TO_STR[i] for i in y_true_enc]
    y_pred = [NUM_TO_STR[i] for i in y_pred_enc]

    microF1 = f1_score(y_true, y_pred, labels=train_labels, average ="micro")
    macroF1 = f1_score(y_true, y_pred, labels=train_labels, average ="macro")

    return {"microF1": microF1, 
            "macroF1": macroF1,
            "y_true": y_true_enc.tolist(),
            "y_pred": y_pred_enc.tolist()}


