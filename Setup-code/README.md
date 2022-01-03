# Report on Transformer experiments

## Setup of the Transformers (SloBERTa)

Desired baseline result (as achieved by Peter): SloBERTa micro F1 0.629, macro F1 0.575


- tried Google Colab and Kaggle, Kaggle seems to be much faster, dataset is saved there, but Google Colab allows nicer collaboration, mounting the Google Drive to the workspace (but data needs to be reloaded every run) -> going with both, prefering Kaggle
- prepared the dataset (see [1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb](https://colab.research.google.com/drive/18bAQjFcVP054bz0_oBszQxjoA_k8fnj8?usp=sharing))
- tried [Peter's demo](https://github.com/TajaKuzman/task5_webgenres/blob/main/Peters-code/Peter-GINCO-demo.ipynb), improved it with elements from his final code grom the task5_webgenres repository (see) -> error in connection with the max_long_seq = 512, 509 (if argument omitted, it works) (see [2-GINCO-SloBERTa-Setup.ipynb](https://colab.research.google.com/drive/1GOyiMOS32VlsIviukAPf9uu13iHK4Lfh?usp=sharing)); deleted instances with no text (result of using only deduplicated paragraphs) (n.a. in dataframe)
- added sliding_window method to Peter's demo ([Copy of 2-GINCO-SloBERTa-Setup-sliding-window.ipnyb](https://colab.research.google.com/drive/1QnSQt25g_Otdyeg43IC3MtLnmtjxhJmj?usp=sharing)) - much higher score, but does not reach the scores from our paper
- tried Peter's final code and prepared FastText texts from his experiments (3-recreating-peters-final-code.ipynb) -> F1 scores higher (microF1: 0.55, macroF1: 0.45), but we have 3-times the examples (instances are repeated 3-times in the FastText experiments) -> I tried Peter's demo again, using 90 epochs. The error regarding max_long_seq is raised here as well. Results improve.
- tried [Peter's demo](https://github.com/TajaKuzman/task5_webgenres/blob/main/Peters-code/Peter-GINCO-demo.ipynb), improved it with his final code from the experiments, ran for 90 epochs, used sliding window (4-joining-demo-and-final-90-epochs-sliding-window.ipynb) 
- repeated the experiment with no sliding window, tried additional things to solve the max_seq_length error (without any success) (5-joining-demo-and-final-90-ep-no-sliding-w.ipynb) -> better results without sliding window

When trying the sliding window method, a warning "Token indices sequence length is longer than the specified maximum sequence length for this model (532 > 512)." suggests that maybe the model uses 512 as max_seq-length either way (??).

Initial results:
- Peter's demo, without the max_long_seq, with sliding_window ([Copy of 2-GINCO-SloBERTa-Setup-sliding-window.ipnyb](https://colab.research.google.com/drive/1QnSQt25g_Otdyeg43IC3MtLnmtjxhJmj?usp=sharing)): in Google Colab training took 2h, 6GB of GPU (60%), in Kaggle it took 17 minutes, F1 (macro, micro) score: 0.409, 0.513 (Kaggle), 0.386, 0.503 (Google Colab)
- Peter's demo, without the max_long_seq, without sliding window, 90 epochs (without sliding window) - improved results: 0.473, 0.533
- Peter's final code from the experiments (on his data - 30 epochs because each instance is repeated 3 times): without max_seq_len: "microF1": 0.55, "macroF1": 0.45; with sliding_window: microF1": 0.52, "macroF1": 0.45
- Peter's demo + final code on my prepared data: without max_seq_len, 90 epochs, deleted instances with no text, sliding_window -> 'microF1': 0.497, 'macroF1': 0.45; confusion matrix not satisfying
- Peter's demo + final code on my prepared data: without max_seq_len, 90 epochs, deleted instances with no text, NO sliding_window -> better results without sliding window: 'microF1': 0.558, 'macroF1': 0.531, better confusion matrix (but: issues with predicting the whole list of texts again - not sure how that impacts the results)
