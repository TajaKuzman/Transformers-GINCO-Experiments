import os
import parse
import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
datadir = "/home/peterr/macocu/task5_webgenres/data/final/fasttext2"

dev_full = os.path.join(datadir, "dev_onlyprimary_True_dedup_False.fasttext")
test_full = os.path.join(datadir, "test_onlyprimary_True_dedup_False.fasttext")
train_full = os.path.join(datadir, "train_onlyprimary_True_dedup_False.fasttext")


dev_dd = os.path.join(datadir, "dev_onlyprimary_True_dedup_True.fasttext")
test_dd = os.path.join(datadir, "test_onlyprimary_True_dedup_True.fasttext")
train_dd = os.path.join(datadir, "train_onlyprimary_True_dedup_True.fasttext")


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
            labels.append(rez["label"])
            texts.append(rez["text"])
        else:
            pass
            #print("error parsing line ", line)
    if encode:
        labels = [STR_TO_NUM[i] for i in labels]
    return pd.DataFrame(data={"text": texts, "labels": labels})

for filename in [train_full, train_dd, test_full, test_dd, dev_full, dev_dd]:
    try:
        _ = parse_fasttext_file(filename)
    except Exception as e:
        raise e


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

def eval_model(test_df):
    y_true_enc = test_df.labels
    y_pred_enc = model.predict(test_df.text.tolist())[0]

    y_true = [NUM_TO_STR[i] for i in y_true_enc]
    y_pred = [NUM_TO_STR[i] for i in y_pred_enc]

    microF1 = f1_score(y_true, y_pred, labels=train_labels, average ="micro")
    macroF1 = f1_score(y_true, y_pred, labels=train_labels, average ="macro")

    return {"microF1": microF1, 
            "macroF1": macroF1,
            "y_true": y_true,
            "y_pred": y_pred}

import pandas as pd

results = list()

with open("backup_12.txt", "r") as f:
    content = f.readline()
    from ast import literal_eval
    content = literal_eval(content)
jsonlikecontent = dict()
for key in content[0].keys():
    jsonlikecontent[key] = [i[key] for i in content]

results = content


train_full_df = parse_fasttext_file(train_full)
test_full_df = parse_fasttext_file(test_full)
dev_full_df = parse_fasttext_file(dev_full)

dev_dd_df = parse_fasttext_file(dev_dd)
test_dd_df = parse_fasttext_file(test_dd)
train_dd_df = parse_fasttext_file(train_dd)

# First experiment: train on full, eval on all available 
for i in range(10):
    print( "First part, run ", i)
    #print(results)
    model = train_model(train_full_df)
    rundict = eval_model(test_full_df)
    rundict["train"] = "full"
    rundict["eval"] = "test_full"
    results.append(rundict)

    rundict = eval_model(dev_full_df)
    rundict["train"] = "full"
    rundict["eval"] = "dev_full"
    results.append(rundict)

    rundict = eval_model(dev_dd_df)
    rundict["train"] = "full"
    rundict["eval"] = "dev_dd"
    results.append(rundict)

    rundict = eval_model(test_dd_df)
    rundict["train"] = "full"
    rundict["eval"] = "test_dd"
    results.append(rundict)
# Second experiment: train on dedup, eval on all available 
for i in range(10):
    print("Run ", i+1, "of 5")
    model = train_model(train_dd_df)
    rundict = eval_model(test_full_df)
    rundict["train"] = "dd"
    rundict["eval"] = "test_full"
    results.append(rundict)

    rundict = eval_model(dev_full_df)
    rundict["train"] = "dd"
    rundict["eval"] = "dev_full"
    results.append(rundict)

    rundict = eval_model(dev_dd_df)
    rundict["train"] = "dd"
    rundict["eval"] = "dev_dd"
    results.append(rundict)

    rundict = eval_model(test_dd_df)
    rundict["train"] = "dd"
    rundict["eval"] = "test_dd"
    results.append(rundict)
import json
with open("backup_12_2.txt", "w") as f:
    json.dump(results, f)