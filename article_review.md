# Article review

Regarding the metrics:

# Beyond the English web:
* They report F1 scores as well as confusion matrices. I was unable to replicate the reported metrics with the provided confusion matrices.
* They measure inter-annotator agreement with F1 scores as well, but they call that a lower bound for classifiers. Isn't it more reasonable to expect the classifier to perform worse than human annotators?
* Splitting is 50-20-30
* 10 registers, later downcast to 7

#  Multilingual and Zero-shot is closing on monolingual web register classification
* CORE dataset, split 50-20-30
* 8 CORE main registers, again downcast to 7
* Again they provide confusion matrices, which perform worse than reported F1 score


#  From bits and numbers to explanations:
+ Precision and recall for 27 most frequent CORE registers, but grouped by lexical, grammatical or both feature sets
+ Agregation again not specified
+ Data split 80-20

#  Exploring the role of lexis and grammar for the stable identification of register in an unrestricted corpus of web documents
* Precision, recall, F1 scores and their standard deviations are reported for combinations of feature types (6 in total, Lexical, Grammatical, combination, word trigram, binary character fourgram....)
* Agregations are not specified, the dataset used is CORE
* Performed 100 experiments to test stability, weighted Welch test is used for p value calculation
* Standard deviations are big, e.g. F1 = 74.54 +/- 15.03
* F1 scores per register are reported in the appendix, graphically for 4 feature sets and in taabular form for 2 feature sets

#  Toward Multilingual Identification of online registers:
* CORE dataset, downsampled to 8 main registers, later only 6 registers results were reported (Lyrical and Spoken were dropped as they were underrepresented in the Finnish data.)
* Per register results reported with AUC scores - to replicate the data directly some preprocessing will be necessary:
```python
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()


lb.fit(y_true)
y_true_bin = lb.transform(y_true)
y_pred_bin = lb.transform(y_pred)
roc_auc_score(y_true_bin, y_pred_bin, average=None)
```


#  Genre Annotation for the Web:
* 10 Functional Text Dimensions (FTDs), precisions and recalls are reported for each, study focuses on precision.
* No splitting, 10-fold cross validation is used
* Confusion matrices are published for Russian and English
* Baseline model: Logistic Regression
* The datasets used is the most similar to ours in size.
* After running the confusion matrix for English, the reported metrics check out.