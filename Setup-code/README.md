# Report on Transformer experiments

## Setting-up the Transformers (SloBERTa)

Desired baseline result (as achieved by Peter): SloBERTa micro F1 0.629, macro F1 0.575

Questions for Peter:
- issues with not loading the test data (the bar stops at 1%, however we get predictions for all examples - does this effect the results?)

Report:

- tried Google Colab and Kaggle, Kaggle seems to be much faster, dataset is saved there, but Google Colab allows nicer collaboration, mounting the Google Drive to the workspace (but data needs to be reloaded every run) -> going with both, prefering Kaggle

- prepared the dataset (see [1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb)


- tried [Peter's demo](https://github.com/TajaKuzman/task5_webgenres/blob/main/Peters-code/Peter-GINCO-demo.ipynb), improved it with elements from his final code grom the task5_webgenres repository ([here](https://github.com/5roop/task5_webgenres)), deleted instances with no text (result of using only deduplicated paragraphs) (n.a. in dataframe) -> error in connection with the max_long_seq = 512 (if argument omitted, it works) (see [2-SloBERTa-Initial-Setup.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/2-SloBERTa-Initial-Setup.ipynb))

When trying the sliding window method, a warning "Token indices sequence length is longer than the specified maximum sequence length for this model (532 > 512)." suggests that maybe the model uses 512 as max_seq-length either way (??).

### Optimal max_seq_length:
Searching for the optimal max_seq_length (code from [2-SloBERTa-Initial-Setup.ipynb](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/blob/main/Setup-code/2-SloBERTa-Initial-Setup.ipynb) - 90 epochs, my prepared data, no sliding_window):
- no parameter: 'microF1': 0.558, 'macroF1': 0.531
- 128
- 200: 'microF1': 0.563, 'macroF1': 0.479
- 250:
- 300

### TO DO:
* add sliding_window parameter to the best performing max_seq_length
* repeat the best performing experiment on Peter's data to see if there are any differences
