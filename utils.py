import os
import json
import parse
import fasttext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

train_labels = ['__label__Legal/Regulation', '__label__Opinionated_News', '__label__News/Reporting', '__label__Forum', '__label__Correspondence', '__label__Invitation', '__label__Instruction', '__label__Recipe', '__label__Opinion/Argumentation', '__label__Promotion_of_Services',
                '__label__Promotion', '__label__List_of_Summaries/Excerpts', '__label__Promotion_of_a_Product', '__label__Call', '__label__Review', '__label__Other', '__label__Information/Explanation', '__label__Interview', '__label__Prose', '__label__Research_Article', '__label__Announcement']
STR_TO_NUM = {s: i for i, s in enumerate(train_labels)}
NUM_TO_STR = {i: s for i, s in enumerate(train_labels)}
NUM_TO_STR_NO_PREFIX = {i: s[9:].replace(
    "_", " ") for i, s in enumerate(train_labels)}
train_labels_no_prefix = [s[9:].replace("_", " ") for s in train_labels]

reduced_labels = ['__label__Legal/Regulation', '__label__Opinionated_News', '__label__News/Reporting', '__label__Forum', '__label__Instruction', '__label__Opinion/Argumentation',
                  '__label__Promotion', '__label__List_of_Summaries/Excerpts', '__label__Other', '__label__Information/Explanation', '__label__Interview', '__label__Announcement']
REDUCED_STR_TO_NUM = {s: i for i, s in enumerate(reduced_labels)}
REDUCED_NUM_TO_STR = {i: s for i, s in enumerate(reduced_labels)}
REDUCED_NUM_TO_STR_NO_PREFIX = {i: s[9:].replace(
    "_", " ") for i, s in enumerate(reduced_labels)}
reduced_labels_no_prefix = [s[9:].replace("_", " ") for s in reduced_labels]


list_of_categories_matrix = ['Information/Explanation', 'Research Article', 'Instruction', 'Recipe', 'Legal/Regulation', 'Call', 'Announcement', 'News/Reporting', 'Opinionated News',
                             'Opinion/Argumentation', 'Review', 'Promotion', 'Promotion of a Product', 'Promotion of Services', 'Invitation', 'Forum', 'Interview', 'Correspondence', 'Prose', 'List of Summaries/Excerpts', 'Other']
list_of_categories_matrix_donwnsampled = ['Information/Explanation', 'Instruction', 'Legal/Regulation', 'Announcement', 'News/Reporting',
                                          'Opinionated News', 'Opinion/Argumentation', 'Promotion', 'Forum', 'Interview', 'List of Summaries/Excerpts', 'Other']


def parse_fasttext_file(path: str, encode=True):
    """Reads fasttext formatted file and returns dataframe.

    Args:
        path (str): fasttext file to be parsed.
        encode (bool, optional): Whether the labels should be encoded to integers. Defaults to True.

    Returns:
        pd.DataFrame: DF with columns `text` and `labels`.
    """
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
    if encode:
        labels = [STR_TO_NUM[i] for i in labels]
    return pd.DataFrame(data={"text": texts, "labels": labels})


def train_model(train_df, NUM_EPOCHS=30, num_labels=21, use_cuda=True, no_cache=True,
                labels=None):
    """Trains a simpletransformer model and returns it.

    Args:
        train_df (pandas.DataFrame): A DataFrame with columns ["text", "labels"].
        NUM_EPOCHS (int, optional): Number of epochs. Defaults to 30.
        num_labels (int, optional): Number of labels used. Defaults to 21.
        use_cuda (bool, optional): Whether to use cuda. Defaults to True. Set False for easier debugging.
        no_cache (bool, optional): Whether to use caching or not. Defaults to True.
        labels (list(str), optional): If not None, use these labels to use string labels instead of numeric labels. 
            Defaults to None. If set, num_labels is calculated automatically.

    Returns:
        simpletransformers.ClassificationModel: a trained model
    """
    from simpletransformers.classification import ClassificationModel, ClassificationArgs

    model_args = ClassificationArgs()
    model_args.num_train_epochs = NUM_EPOCHS
    model_args.learning_rate = 1e-5
    model_args.overwrite_output_dir = True
    model_args.train_batch_size = 32
    model_args.no_cache = no_cache
    model_args.no_save = True
    model_args.save_steps = -1
    model_args.max_seq_length = 512
    model_args.silent = True
    if labels:
        LABELS = list(LABELS)
        model_args.labels_list = LABELS
        num_labels = len(LABELS)

    model = ClassificationModel(
        "camembert", "EMBEDDIA/sloberta",
        num_labels=num_labels,
        use_cuda=use_cuda,
        args=model_args
    )
    model.train_model(train_df)
    return model


def eval_model(test_df, model):
    """Evaluates trained model on test_df and returns metrics.

    Args:
        test_df (pd.DataFrame): dataframe with `text` and `labels` columns
        model (simpletransformers.ClassificationModel): previously trained model to evaluate.

    Returns:
        results (dict): dictionary with fields `microF1`, `macroF1`, `y_true`, `y_pred`.
    """
    y_true_enc = test_df.labels
    y_pred_enc = model.predict(test_df.text.tolist())[0]

    y_true = [NUM_TO_STR[i] for i in y_true_enc]
    y_pred = [NUM_TO_STR[i] for i in y_pred_enc]

    microF1 = f1_score(y_true, y_pred, labels=train_labels, average="micro")
    macroF1 = f1_score(y_true, y_pred, labels=train_labels, average="macro")

    return {"microF1": microF1,
            "macroF1": macroF1,
            "y_true": y_true_enc.tolist(),
            "y_pred": y_pred_enc.tolist()}


def plot_cm(y_true, y_pred,  save=False, title=None, labels=None,
            include_metrics=True, figsize=None):
    """Plots confusion matrix for y_true and y_pred. Can calculate
    metrics and display them in the title.

    Args:
        y_true (list|np.array|pd.Series): true labels
        y_pred (list|np.array|pd.Series): predicted labels
        save (bool|str, optional): string, path to save a figure to, or bool (False) to not save. Defaults to False.
        title ([None|str], optional): Title to put at the top of the figure. Defaults to None.
        labels (list|np.array|pd.Series, optional): how to arrange labels. The list must contain all labels
            that appear in y_true and y_pred. Defaults to None.
        include_metrics (bool, optional): Whether the metrics should be displayed
            under the title. Defaults to True.
        figsize (tuple(int, int), optional): figure size. Defaults to None.

    Returns:
        microF1, macroF1: metrics.
    """            
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    import matplotlib.pyplot as plt
    if not labels:
        labels = list_of_categories_matrix
    plt.style.use(["science", "no-latex", ])
    cm = confusion_matrix(y_true, y_pred, labels=labels, )
    cm = cm/3
    cm = cm.astype(int)
    # print(cm)
    if figsize == None:
        figsize = (9, 9)
    plt.figure(figsize=figsize)
    plt.imshow(cm, cmap="Oranges")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, '{:d}'.format(z), ha='center', va='center')
    classNames = labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=90)
    plt.yticks(tick_marks, classNames)
    microF1 = f1_score(y_true, y_pred, labels=labels, average="micro")
    macroF1 = f1_score(y_true, y_pred, labels=labels, average="macro")

    print(f"{microF1=:0.4}")
    print(f"{macroF1=:0.4}")

    metrics = f"{microF1=:0.4}, {macroF1=:0.4}" if include_metrics else ""
    if title:
        if include_metrics:
            plt.title(title + ";\n" + metrics)
        else:
            plt.title(title)
    else:
        plt.title(metrics)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    return microF1, macroF1


def read_record(filename: str) -> pd.DataFrame:
    """Reads a record file and returns a DataFrame.

    Args:
        filename (str): record filename

    Returns:
        pd.DataFrame: resulting dataframe
    """
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


def downsample_second(numlabel: int) -> int:
    """Performs the downsampling based on the second downsampling.

    Args:
        numlabel (int): numerical label, original

    Returns:
        int: downsampled numerical label
    """
    stringlabel = NUM_TO_STR[numlabel]
    second_original = {"Recipe": "Instruction", "Research Article": "Information/Explanation", "Review": "Opinion/Argumentation", "Promotion of Services": "Promotion",
                       "Promotion of a Product": "Promotion", "Invitation": "Promotion", "Correspondence": "Other", "Prose": "Other", "Call": "Other"}

    second = {
        f"__label__{k.replace(' ', '_')}": f"__label__{v.replace(' ', '_')}" for k, v in second_original.items()}

    new_stringlabel = second.get(stringlabel, stringlabel)

    return REDUCED_STR_TO_NUM[new_stringlabel]


def to_label(l: list, reduced=False) -> list:
    """Transform a series of strings or integers into original labels.

    Args:
        l (list): list or pandas.Series with either string or numeric labels.
        reduced (bool, optional): if True, use reduced label set (12 labels). Defaults to False.

    Raises:
        AttributeError: there seems to be a weird input type

    Returns:
        list: list of labels with no prefix
    """    
    
    to_return = list()
    for i in l:
        if type(i) == int:
            if reduced:
                to_return.append(REDUCED_NUM_TO_STR_NO_PREFIX[i])
            else:
                to_return.append(NUM_TO_STR_NO_PREFIX[i])
        elif type(i) == str:
            to_return.append(i[9:].replace("_", " "))
        else:
            raise AttributeError(f"Got type {type(i)} for input {i}")
    return to_return


def train_model_xlm_roberta(train_df, NUM_EPOCHS=30, num_labels=21, use_cuda=True, no_cache=True):
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
        "xlm-roberta", "xlm-roberta-base",
        num_labels=num_labels,
        use_cuda=use_cuda,
        args=model_args
    )
    model.train_model(train_df)
    return model
