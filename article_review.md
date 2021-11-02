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
  
# Addendum 2021-11-02T10:59:10

I once more replicated the confusion matrix from _Beyond The English web_, but this time I included data from Table 1, namely distribution of data instances in the corpus. I looked at the Finnish example. In the normalized confusion matrix I multiplied the rows with the percentages of the registers.

 >E.g.:
 >register HI (Howto/Instruction) had a true positive rate of 0.62, and the HI register represents 6.47% of the corpus.
 >I then multiplied 62 * 647 together and generated 40114 instances with this label and added them to true and predicted labels.
 >27% of the HI register was labeled as IN, so again I generated 27 * 647 IN labels and added them to predicted labels , and 27 * 647 HI labels were added to true labels.

The F1 scores reported were 76.28% on dev and 73.18% on test data. We only know that the confusion matrices are plotted for the best performing model, but we do not know explicitly which split was used. For the plotted Finnish-Finnish confusion matrix I calculated

```
average='micro' F1 score: 0.7227684201225731
average='macro' F1 score: 0.600648043795533
```

None of these agree precisely with the data reported, but we might postulate that a rounding error could account for this and the metric used was in fact micro F1.

In the Swedish - Swedish example we expect to get either 82.61% or 83.04%, but instead get:
```
average='micro' F1 score: 0.7566619480579286
average='macro' F1 score: 0.6750929173692294
```

This means that we still do not know which F1 is reported.
