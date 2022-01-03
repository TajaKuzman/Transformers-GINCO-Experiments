# Report on Transformer experiments

## Setting-up the Transformers (SloBERTa)

Desired baseline result (as achieved by Peter): SloBERTa micro F1 0.629, macro F1 0.575 -> achieved and outperformed with max_seq_length 300

### Report

- tried Google Colab and Kaggle, Kaggle seems to be much faster, dataset is saved there, but Google Colab allows nicer collaboration, mounting the Google Drive to the workspace (but data needs to be reloaded every run) -> going with both, prefering Kaggle

- prepared the dataset (see [1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb))


- tried [Peter's demo](https://github.com/TajaKuzman/task5_webgenres/blob/main/Peters-code/Peter-GINCO-demo.ipynb), improved it with elements from his final code grom the task5_webgenres repository ([here](https://github.com/5roop/task5_webgenres)), deleted instances with no text (result of using only deduplicated paragraphs) (n.a. in dataframe) -> error in connection with the max_long_seq = 512 (if argument omitted, it works) (see [2-SloBERTa-Initial-Setup.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/2-SloBERTa-Initial-Setup.ipynb))

- tried adding the sliding_window parameter - worse performance, training takes much more time -> further experiments without the sliding_window

### Data Preparation
- deduplicated text (based on our previous research)
- primary label only, categories with less than 5 instances merged to Other -> 21 labels (primary label level 2)
- used train + dev split as train split (no hyperparameter optimisation planned), tested on the test split (802:200 texts)
- 19 empty instances (with no text - result of using only non-duplicated text from the instances) removed -> 786:197 (train:test) texts

### Optimal max_seq_length:
Searching for the optimal max_seq_length (code from [2-SloBERTa-Initial-Setup.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/2-SloBERTa-Initial-Setup.ipynb) - 90 epochs, my prepared data (but only train data, not train+dev), no sliding_window, ! just one run for each setup):
- no parameter: 'microF1': 0.558, 'macroF1': 0.531; Peter's data (3-recreating-peters-final-code.ipynb - 30 epochs, instances repeated 3 times): 'macroF1': 0.514, 'microF1': 0.584
- 128: 'microF1': 0.523, 'macroF1': 0.460
- 200: 'microF1': 0.563, 'macroF1': 0.479
- 250: 'microF1': 0.584, 'macroF1': 0.488
- 300: 'microF1': 0.665, 'macroF1': 0.595
- 350: 'microF1': 0.604, 'macroF1': 0.556
- 390: 'microF1': 0.599, 'macroF1': 0.531
- from 400 onwards -> error


### TO DO:
- ugotovi, kakšen je rezultat pri 300 max_seq_length pri train+dev: Macro f1: 0.567, Micro f1: 0.64 (rahlo slabši)
- poskusi: znižaj train batch size (min. 21), poskusi če potem dela s 512 - ne dela; 400 dela, traja dlje -> ostanimo kar pri batch size 32, da ne spreminjamo preveč
- ugotovi, kakšen je najmanjši seq_length, da vsi modeli lavfajo