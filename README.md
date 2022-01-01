# Report on Transformer experiments

## Setup of the Transformers (SloBERTa)

- tried Google Colab and Kaggle, Kaggle seems to be much faster, but Google Colab allows nicer collaboration, mounting the Google Drive to the workspace
- prepared the dataset (see [1-GINCO-and-Transformers_initial_experiments_with_the_code.ipynb](https://colab.research.google.com/drive/18bAQjFcVP054bz0_oBszQxjoA_k8fnj8?usp=sharing))
- tried Peter's demo (but used 30 epochs, not 90) -> error in connection with the max_long_seq = 512, 509 (if argument omitted, it works) (see [2-GINCO-SloBERTa-Setup.ipynb](https://colab.research.google.com/drive/1GOyiMOS32VlsIviukAPf9uu13iHK4Lfh?usp=sharing)); errors in predicting the test_df as a list -> loop for predicting each instance; deleted instances with no text (result of using only deduplicated paragraphs) (n.a. in dataframe)
- added sliding_window method to Peter's demo ([Copy of 2-GINCO-SloBERTa-Setup-sliding-window.ipnyb](https://colab.research.google.com/drive/1QnSQt25g_Otdyeg43IC3MtLnmtjxhJmj?usp=sharing)) - much higher score, but does not reach the scores from our paper

When trying the sliding window method, a warning "Token indices sequence length is longer than the specified maximum sequence length for this model (532 > 512)." suggests that maybe the model uses 512 as max_long_seq either way (??).

Initial results:
- Peter's demo, without the max_long_seq: F1 score: macro 0.24-0.27, micro 0.44-0.495 ([2-GINCO-SloBERTa-Setup.ipynb](https://colab.research.google.com/drive/1GOyiMOS32VlsIviukAPf9uu13iHK4Lfh?usp=sharing))
- Peter's demo, without the max_long_seq, with sliding_window ([Copy of 2-GINCO-SloBERTa-Setup-sliding-window.ipnyb](https://colab.research.google.com/drive/1QnSQt25g_Otdyeg43IC3MtLnmtjxhJmj?usp=sharing)): in Google Colab training took 2h, 6GB of GPU (60%), in Kaggle it took 17 minutes, F1 (macro, micro) score: 0.409, 0.513 (Kaggle), 0.386, 0.503 (Google Colab)