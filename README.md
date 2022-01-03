# Report on Transformer experiments

## The task

I would like to perform text classification experiments, more specifically automated (web) genre identification. I would perform the classification experiments with the Transformer-based pre-trained language models on the Slovene web genre identification corpus GINCO. More specifically, I am interested in comparing the performance of the following monolingual and multilingual Transformer models on this task:

* the Slovene SloBERTa model
* Slovene-Croatian-English CroSloEngual BERT model
* the multilingual models XLM-RoBERTa and DeBERTaV3
* the model for related South Slavic languages BERTić
* the monolingual English BERT base model (cased)
* comparison of the size: the base-sized and the large-sized XML-RoBERTa


## Setting-up the Transformers (SloBERTa)

- read [README.md](https://github.com/TajaKuzman/Transformers-GINCO-Experiments/tree/main/Setup-code) in the Setup-code folder