# task5_webgenres

## Data preparation

I performed the splitting into train, dev and test splits, as documented in `1_data_preparation.ipynb`. In this step I only took into account stratification by year. It turns out that the ratios of hardness is preserved in this case, but the ratios of primary labels across folds is not preserved as much as we would perhaps hope. I attempt to correct this in `1a_data_preparation.ipynb`. I tried to split it by stratifying crawl year, primary labels and hardness, but it turned out that this is not possible, as some combinations of these parameters only had one instance.

I therefore discarded hardness from the stratification process and opted only for crawl date and primary label. I performed the same data leakage corrections as before. Below I'm presenting a few distributions for both scenarios:

### Crawl date:

Original:

![](.images/../images/1_crawled.png)

After including primary labels:

![](images/1b_crawled.png)

### Hardness:

Original: 

![](images/1_hardness.png)

After including primary labels:

![](images/1b_hardness.png)

### Primary labels:

Original:

![](images/1_primaries.png)

After including primary labels:

![](images/1b_primaries.png)

As we can see, the distribution of the primary labels has changed more with the 
