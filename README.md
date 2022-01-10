# Report on Transformer experiments

## The task

I will perform the classification experiments with the Transformer-based pre-trained language models on the Slovene web genre identification corpus GINCO. More specifically, I am interested in comparing the performance of the following monolingual and multilingual Transformer models on this task:

* the Slovene SloBERTa model
* Slovene-Croatian-English CroSloEngual BERT model
* the multilingual models XLM-RoBERTa and DeBERTaV3
* the model for related South Slavic languages BERTić
* the monolingual English BERT base model (cased)
* comparison of the size: the base-sized and the large-sized XML-RoBERTa

## Preparing the dataset
The data, used for the experiments, is located in the *data* folder. I used *GINCO_dataframe_dedup_train_dev.csv* (train + dev split) as training data and *GINCO_dataframe_dedup_test.csv* as test data. The data was prepared from the "GINCO-1.0-suitable.json" file according to the code in [1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb).

## Setting-up the Transformers (SloBERTa)

- read [README.md](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/tree/main/Setup-code) in the Setup-code folder.

## Training the Transformers

- see the notebook [*6-transformer-comparison-2-same-parameters.ipynb*](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/6-transformer-comparison-2-same-parameters.ipynb)

Baseline result (as achieved by Peter in our previous work): SloBERTa micro F1 0.629, macro F1 0.575 -> to follow the previous experiments as closely as possible, I used the same hyperparameters, except:
- "num_train_epochs": 90 instead of 30 (in his data, each instance is repeated 3 times to accomodate some additional research questions, in our experiments, that is not needed -> the number of training epochs is thus 3 times bigger in our experiments to get similar results)
- "max_seq_length": 128 instead of 512 (available GPU memory size constraint in Kaggle - the large-sized model does not work with the parameter set to 512; early experiments were done with max_seq_length 300 which worked for all models except the large-sized one - see [*4-Transformer-comparison-setup.ipynb*](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/4-Transformer-comparison-setup.ipynb))
- "train_batch_size": 21 (GPU memory size constraint for the large-sized model; earlier experiments in *4-Transformer-comparison-setup.ipynb* used batch size 32)

The hyperparameters:

`
model_args ={"overwrite_output_dir": True,
             "num_train_epochs": 90,
             "labels_list": LABELS,
             "learning_rate": 1e-5,
             "train_batch_size": 21,
             "no_cache": True,
             "no_save": True,
             "max_seq_length": 128,
             "save_steps": -1,
             }
`

The mDeBERTaV3 model uses the model type debertav2, which is [not yet supported](https://simpletransformers.ai/docs/classification-specifics/) by the Simple Transformers, and training on it is not successful (it assigns all the instances to one label), so the model had to be omitted from the comparison.

Each model was ran 5 times to be able to analyse the statistical importance of the differences between them.


## Results

Materials for analysing results are located in the *results* folder:
* the code for creating tables, plots, confusion matrices and calculating statistical significance: [*Analysing_results.ipynb*](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/results/Analysing_results.ipynb)
* JSON files with the F1 scores, and lists of true and predicted labels for each run

Confusion matrices and scatterplots are located in the *plots* folder.