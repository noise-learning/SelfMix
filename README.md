# SelfMix: Robust Learning Against Textual Label Noise with Self-Mixup Training

## Abstract

The conventional success of textual classification relies on annotated data, and the new paradigm of pre-trained language models (PLMs) still requires a few labeled data for downstream tasks. However, in real-world applications, label noise inevitably exists in training data, damaging the effectiveness, robustness, and generalization of the models constructed on such data. Recently, remarkable achievements have been made to mitigate this dilemma in visual data, while only a few explore textual data. To fill this gap, we present SelfMix, a simple yet effective method, to handle label noise in text classification tasks. SelfMix uses the Gaussian Mixture Model to separate samples and leverages semi-supervised learning. Unlike previous works requiring multiple models, our method utilizes the dropout mechanism on a single model to reduce the confirmation bias in self-training and introduces a textual level mixup training strategy. Experimental results on three text classification benchmarks with different types of text show that the performance of our proposed method outperforms these strong baselines designed for both textual and visual data under different noise ratios and noise types.

## Illustration

![framework](img/framework.png)

## Environment

1. python 3.6
2. torch
3. transformers
4. sklearn

## Experiments

We do experiments on three text classification benchmarks of different types, including Trec: a question-type dataset, AG-News: a news categorization dataset, and IMDB: a sentiment analysis dataset. The folder already has the Trec dataset to test, and you can download others yourself. Please process label to id (start from 0).

- For symmetric or asymmetric noise, we generate noisy labels by 'filp_label' in read_data.py, and you can run the model directly through

```bash
bash train_asym.sh
```

For different datasets, we recommend using hyperparameters as follows.

| Dataset  | Trec | AG-News | IMDB |
| -------- | ---- | ------- | ---- |
| lambda_p | 0.2  | 0.2     | 0.1  |
| lambda_r | 0.3  | 0.3     | 0.5  |

- For instance-dependent noise (IDN), following [Algan and Ulusoy, 2020](https://arxiv.org/pdf/2003.10471.pdf), we train an LSTM classifier on a small set of the original training data and flip the origin labels to the class with the highest prediction probability among other classes. We keep the hyperparameters ($\lambda_p$, $\lambda_r$) as (0.0, 0.3) constant for all datasets under instance-dependent noise, and you can run the model directly through

```bash
bash train_idn.sh
```

