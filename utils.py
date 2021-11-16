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
NUM_TO_STR_NO_PREFIX = {i: s[9:].replace("_", " ") for i, s in enumerate(train_labels)}
train_labels_no_prefix = [s[9:].replace("_", " ") for s in train_labels]

reduced_labels = ['__label__Legal/Regulation', '__label__Opinionated_News', '__label__News/Reporting', '__label__Forum', '__label__Instruction', '__label__Opinion/Argumentation', '__label__Promotion', '__label__List_of_Summaries/Excerpts', '__label__Other', '__label__Information/Explanation', '__label__Interview', '__label__Announcement']
REDUCED_STR_TO_NUM = {s: i for i, s in enumerate(reduced_labels)}
REDUCED_NUM_TO_STR = {i: s for i, s in enumerate(reduced_labels)}
REDUCED_NUM_TO_STR_NO_PREFIX = {i: s[9:].replace("_", " ") for i, s in enumerate(reduced_labels)}
reduced_labels_no_prefix = [s[9:].replace("_", " ") for s in reduced_labels]

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



def train_model(train_df, NUM_EPOCHS=30, num_labels=21, use_cuda=True, no_cache=True):
    from simpletransformers.classification import ClassificationModel
    model_args = {
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": 1e-5,
        "overwrite_output_dir": True,
        "train_batch_size": 32,
        "no_save": True,
        "no_cache": no_cache,
        "overwrite_output_dir": True,
        "save_steps": -1,
        "max_seq_length": 512,
        "silent": True
    }

    model = ClassificationModel(
        "camembert", "EMBEDDIA/sloberta",
        num_labels = num_labels,
        use_cuda = use_cuda,
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



def plot_cm(y_true, y_pred,  save=False, title=None, labels=None):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt
    if not labels:
        labels = train_labels
    plt.style.use(["science", "no-latex", ])
    cm = confusion_matrix(y_true, y_pred, labels=labels, )
    cm = cm/3
    # print(cm)
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, cmap="Oranges")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
    classNames = labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=90)
    plt.yticks(tick_marks, classNames)
    microF1 = f1_score(y_true, y_pred, labels=labels, average ="micro")
    macroF1 = f1_score(y_true, y_pred, labels=labels, average ="macro")

    print(f"{microF1=:0.4}")
    print(f"{macroF1=:0.4}")

    metrics = f"{microF1=:0.4}, {macroF1=:0.4}"
    if title:
        plt.title(title +";\n" + metrics)
    else:
        plt.title(metrics)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    return microF1, macroF1

def read_record(filename: str) -> pd.DataFrame:
    import json
    import pandas as pd
    pd.set_option("precision", 3)
    with open(filename) as f:
        content = json.load(f)
    jsonlikecontent = dict()
    for key in content[0].keys():
        jsonlikecontent[key] = [i[key] for i in content]
    df = pd.DataFrame(data=jsonlikecontent)
    return df


def downsample_second(numlabel):
    stringlabel = NUM_TO_STR[numlabel]
    second_original = {"Recipe":"Instruction", "Research Article":"Information/Explanation", "Review":"Opinion/Argumentation", "Promotion of Services":"Promotion", "Promotion of a Product":"Promotion", "Invitation":"Promotion", "Correspondence":"Other", "Prose":"Other", "Call":"Other"}

    second = {f"__label__{k.replace(' ', '_')}": f"__label__{v.replace(' ', '_')}" for k, v in second_original.items()}

    new_stringlabel = second.get(stringlabel, stringlabel)

    return REDUCED_STR_TO_NUM[new_stringlabel]


def to_label(l:list, reduced=False) -> list:
    """To be applied on whole pandas series.
    Returns a prettified label list, regardless of input is a list of 
    integers or a list of strings.
    
    If reduced labels are used, set reduced=True."""
    to_return = list()
    for i in l:
        if type(i) == int:
            if reduced:
                # stringlabel = REDUCED_NUM_TO_STR[i]
                # to_return.append(stringlabel[9:].replace("_", " "))
                to_return.append(REDUCED_NUM_TO_STR_NO_PREFIX[i])
            else:
                to_return.append(NUM_TO_STR_NO_PREFIX[i])
        elif type(i) == str:
            to_return.append(i[9:].replace("_", " "))
        else:
            raise AttributeError(f"Got type {type(i)} for input {i}")
    return to_return