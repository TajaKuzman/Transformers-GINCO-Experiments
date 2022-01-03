# Report on Transformer experiments

## Setting-up the Transformers (SloBERTa)

### Report

- tried Google Colab and Kaggle, Kaggle seems to be much faster, dataset is saved there, but Google Colab allows nicer collaboration, mounting the Google Drive to the workspace (but data needs to be reloaded every run) -> going with both, prefering Kaggle

- prepared the dataset (see [1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb))


- tried [Peter's demo](https://github.com/TajaKuzman/task5_webgenres/blob/main/Peters-code/Peter-GINCO-demo.ipynb), improved it with elements from his final code grom the task5_webgenres repository ([here](https://github.com/5roop/task5_webgenres)), deleted instances with no text (result of using only deduplicated paragraphs) (n.a. in dataframe) -> error in connection with the max_long_seq = 512 (if argument omitted, it works) (see [2-SloBERTa-Initial-Setup.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/2-SloBERTa-Initial-Setup.ipynb))

- tried adding the sliding_window parameter - worse performance, training takes much more time -> further experiments without the sliding_window

- tried lowering the batch size (to 21) to be able to use max_seq_length=512 -> still error

### Data Preparation
- deduplicated text (based on our previous research)
- primary label only, categories with less than 5 instances merged to Other -> 21 labels (primary label level 2)
- used train + dev split as train split (no hyperparameter optimisation planned), tested on the test split (802:200 texts)
- 19 empty instances (with no text - result of using only non-duplicated text from the instances) removed -> 786:197 (train:test) texts

### Optimal max_seq_length:
Since the 512 as max_seq_length does not work on Kaggle due to the GPU memory restriction, I searched for the optimal max_seq_length (code from [2-SloBERTa-Initial-Setup.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/2-SloBERTa-Initial-Setup.ipynb) - 90 epochs, my prepared data (but only train data, not train+dev), no sliding_window, ! just one run for each setup):
- no parameter: 'microF1': 0.558, 'macroF1': 0.531
- 128: 'microF1': 0.523, 'macroF1': 0.460
- 200: 'microF1': 0.563, 'macroF1': 0.479
- 250: 'microF1': 0.584, 'macroF1': 0.488
- 300: 'microF1': 0.665, 'macroF1': 0.595
- 350: 'microF1': 0.604, 'macroF1': 0.556
- 390: 'microF1': 0.599, 'macroF1': 0.531
- from 400 onwards -> error

**Conclusion**: Although these results are from only one run per setup, it seems safe to say that increasing the parameter improves the results. As 400 results in an error, the next step before determining the size of this parameter is to see whether all other Transformer models allow the max_seq_length of 300.

### Optimal max_seq_length for all models to work:
- testing the code from 4-transformer-comparison-setup.ipynb; testing whether max_seq_length=300 works on all models
- this size works with the following models: SloBERTa, CroSloEngBERT, XML-RoBERTa (base-sized), deBERTaV3, BERTić and English BERT
- it does not work with the large-sized XML-RoBERTa, the model works only with max_seq_length =< 128 (default), with a smaller size of batches - 21 (on 32 batches does not work)

**Conclusion**: To compare monolingual and multilingual models, I will use the max_seq_length=300. When analysing the impact of the size of the model (comparison of base-sized and large-sized XML-RoBERTa), I will use the setup under which the larger model works for both models: max_seq_length = 128, train_batch_size: 21