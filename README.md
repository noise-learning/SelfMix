## SelfMix: Robust Learning Against Textual Label Noise with Self-Mixup Training

This repository contains the code and pre-trained models for our paper [SelfMix: Robust Learning Against Textual Label Noise with Self-Mixup Training]()


## Overview

We proposes SelfMix, i.e., a self-distillation robust training method based on the pre-trained models.

SelfMix uses GMM to select the samples that are more likely to be wrong and erase their original labels. Then we leverage semi-supervised learning to jointly train the labeled set `X` (contains mostly clean samples) and an unlabeled set `U` (contains mostly noisy samples).

![](figure/model.png)

## Datasets

We do experiments on three text classification benchmarks of different types, including Trec, AG-News and IMDB.

| Dataset | Class | Type | Train | Test |
|:--------|:-----:|:----:|:-----:|:-----|
|  Trec | 6 | Question-Type | 5452 | 500 |
| IMDB | 2 | Sentiment Analysis | 45K | 5K |
| AG-News | 4 | News Categorization | 120K | 7.6K |


### Noise Sample Generation

We evaluate our strategy under the following two types of label noise

* Asymmetric noise (Asym): Following [Chen et al.](https://arxiv.org/abs/1905.05040), we choose a certain proportion of samples and flip their labels to the corresponding class according to the asymmetric noise transition matrix.
* Instance-dependent Noise (IDN): Following [Algan and Ulusoy](https://arxiv.org/abs/2003.10471), we train an LSTM classifier on a small set of the original training data and flip the origin labels to the class with the highest prediction.

You can construct noisy datsets by the following command (e.g., Trec 0.4asym),

```bash
python data/corrupt.py \
    --src_data_path data/trec/train.csv \
    --save_path data/trec/train_corrupted.csv \
    --noise_type asym \
    --noise_ratio 0.4
```

Since generating IDN is a bit more complex, we provide datasets of our experiments directly [here]().

### Hyperparameters

We use the following hyperparamters for training SelfMix:

| Data Settings | Trec/AG-News(Asym) | IMDB(Asym) | AG-News/IMDB(IDN) |
|:--------|:-----:|:----:|:-----:|
| `lambda_p` | 0.2 | 0.1 | 0.0 |
| `lambda_r` | 0.3 | 0.5 | 0.3 |
| `class_reg` | False | False | True |

## Train

In the following section, we describe how to train a SelfMix model by using our code.

### Requirements

You should run the following script to install the remaining dependencies first.

```bash
pip install -r requirements.txt
```

### Quick Start

We list some demo config in folder `demo_config`, you can just use the demo config to train,

```bash
python train.py demo_config/trec-bert-asym_train.json
```

### Parameters

Details about the meaning of parameters can be seen in our paper and dataclass `ModelArguments`, `DataTrainingArguments` and `OurTrainingArguments` in `train.py`

## Evaluation

Similarly, you can run evaluation by the following command,

```bash
python evaluation.py demo_config/trec-bert_eval.json
```

Details about parameters can be seen in dataclass `ModelArguments` and `DataEvalArguments` in `evaluation.py`.
