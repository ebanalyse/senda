# SENDA <img src="logo.png" align="right" height=250/>

![Build status](https://github.com/ebanalyse/NERDA/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/ebanalyse/NERDA/branch/main/graph/badge.svg?token=OB6LGFQZYX)](https://codecov.io/gh/ebanalyse/NERDA)
![PyPI](https://img.shields.io/pypi/v/NERDA.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/NERDA?color=green)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

`SENDA` is a python package for fine-tuning transformers for 
sentiment analysis.

`SENDA` builds on the excellent `transformers.Trainer` API.

## Installation guide
`SENDA` can be installed from [PyPI](https://pypi.org/project/SENDA/) with 

```
pip install SENDA
```

If you want the development version then install directly from [GitHub](https://github.com/ebanalyse/SENDA).

## How to use

We will fine-tune a transformer for detecting the polarity ('positive', 'neutral' or 'negative')
of Danish Tweets. We build the model on more than 5,000 Danish Tweets kindly annotated
and hosted by the Alexandra Institute.

First, load the datasets, we want to use for fine-tuning our sentiment analysis model.

```python
from SENDA import get_danish_tweets
df_train, df_eval, df_test = get_danish_tweets()
```
Note, that the datasets must be DataFrames, that contains the columns 'text' and 'label'.

Next, instantiate the model and set up the model.

```python
from SENDA import Model
m = Model(train_dataset = df_train, 
          eval_dataset = df_eval,
          transformer = "Maltehb/danish-bert-botxo",
          labels = ['negativ', 'neutral', 'positiv'],
          tokenize_args = {'padding':True, 'truncation':True, 'max_length':512},
          training_args = {"output_dir":'./results',          # output directory
                           "num_train_epochs": 4,              # total # of training epochs
                           "per_device_train_batch_size":8,  # batch size per device during training
                           "evaluation_strategy":"steps",
                           "eval_steps":100,
                           "logging_steps":100,
                           "learning_rate":2e-05,
                           "weight_decay": 0.01,
                           "per_device_eval_batch_size":32,   # batch size for evaluation
                           "warmup_steps":100,                # number of warmup steps for learning rate scheduler
                           "seed":42,
                           "load_best_model_at_end":True,
                           })
```

Now, all there is left is initialize a `transformers.Trainer` and 
train the model:

```python
# initialize Trainer
m.init()
# run training
m.train()
```

The model can then be evaluated on the test set:

```python
m.evaluate(df_test)
```

You can predict new observations by:

```python
text = "Sikke en dejlig dag det er i dag"
# in English: 'What a lovely day'
m.predict(text, return_labels=True)
```

### Model Performance

The table below summarizes the performance (F1-scores) of the precooked `SENDA` models.

| **Level**     | `DA_BERT_ML` | `DA_ELECTRA_DA` | `EN_BERT_ML` | `EN_ELECTRA_EN` |
|---------------|--------------|-----------------|--------------|-----------------|
| B-PER         | 93.8         | 92.0            | 96.0         | 95.1            |      
| I-PER         | 97.8         | 97.1            | 98.5         | 97.9            |   
| B-ORG         | 69.5         | 66.9            | 88.4         | 86.2            |     
| I-ORG         | 69.9         | 70.7            | 85.7         | 83.1            |   
| B-LOC         | 82.5         | 79.0            | 92.3         | 91.1            |     
| I-LOC         | 31.6         | 44.4            | 83.9         | 80.5            |     
| B-MISC        | 73.4         | 68.6            | 81.8         | 80.1            |     
| I-MISC        | 86.1         | 63.6            | 63.4         | 68.4            |   
| **AVG_MICRO** | 82.8         | 79.8            | 90.4         | 89.1            |      
| **AVG_MACRO** | 75.6         | 72.8            | 86.3         | 85.3            |

## Background
`SENDA` is developed as a part of [Ekstra Bladet](https://ekstrabladet.dk/)’s activities on Platform Intelligence in News (PIN). PIN is an industrial research project that is carried out in collaboration between the [Technical University of Denmark](https://www.dtu.dk/), [University of Copenhagen](https://www.ku.dk/) and [Copenhagen Business School](https://www.cbs.dk/) with funding from [Innovation Fund Denmark](https://innovationsfonden.dk/). The project runs from 2020-2023 and develops recommender systems and natural language processing systems geared for news publishing, some of which are open sourced like `SENDA`.

## Contact
We hope, that you will find `SENDA` useful.

Please direct any questions and feedbacks to
[us](mailto:lars.kjeldgaard@eb.dk)!

If you want to contribute (which we encourage you to), open a
[PR](https://github.com/ebanalyse/SENDA/pulls).

If you encounter a bug or want to suggest an enhancement, please 
[open an issue](https://github.com/ebanalyse/SENDA/issues).