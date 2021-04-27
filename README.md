# senda <img src="logo.png" align="right" height=250/>

![Build status](https://github.com/ebanalyse/NERDA/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/ebanalyse/NERDA/branch/main/graph/badge.svg?token=OB6LGFQZYX)](https://codecov.io/gh/ebanalyse/NERDA)
![PyPI](https://img.shields.io/pypi/v/NERDA.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/NERDA?color=green)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

`senda` is a python package for fine-tuning transformers for 
sentiment analysis (and text classification in general).

`senda` builds on the excellent `transformers.Trainer` API.

## Installation guide
`senda` can be installed from [PyPI](https://pypi.org/project/senda/) with 

```
pip install senda
```

If you want the development version then install directly from [GitHub](https://github.com/ebanalyse/senda).

## How to use

We will fine-tune a transformer for detecting the polarity ('positive', 'neutral' or 'negative')
of Danish Tweets. For training we use more than 5,000 Danish Tweets kindly annotated
and hosted by the Alexandra Institute.

First, load the datasets, we want to use for fine-tuning our sentiment analysis model.

```python
from senda import get_danish_tweets
df_train, df_eval, df_test = get_danish_tweets()
```
Note, that the datasets must be DataFrames containing the columns 'text' and 'label'.

Next, instantiate the model and set up the model.

```python
from senda import Model
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

Now, all there is left is to initialize a `transformers.Trainer` and 
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

Predict new observations:

```python
text = "Sikke en dejlig dag det er i dag"
# in English: 'What a lovely day'
m.predict(text, return_labels=True)
```

### senda model available on Huggingface

The model above achieves an accuracy of 0.76 and a macro-averaged F1-score of 0.75 on a small test data set, that Alexandra Institute provides.

The model is published on [Huggingface](https://huggingface.co/larskjeldgaard/senda).

Here is how to download and use the model with PyTorch:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("larskjeldgaard/senda")
model = AutoModelForSequenceClassification.from_pretrained("larskjeldgaard/senda")

# create 'senda' sentiment analysis pipeline 
senda_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

senda_pipeline("Sikke en dejlig dag det er i dag")
```

## Background
`senda` is developed as a part of [Ekstra Bladet](https://ekstrabladet.dk/)’s activities on Platform Intelligence in News (PIN). PIN is an industrial research project that is carried out in collaboration between the [Technical University of Denmark](https://www.dtu.dk/), [University of Copenhagen](https://www.ku.dk/) and [Copenhagen Business School](https://www.cbs.dk/) with funding from [Innovation Fund Denmark](https://innovationsfonden.dk/). The project runs from 2020-2023 and develops recommender systems and natural language processing systems geared for news publishing, some of which are open sourced like `senda`.

## Contact
We hope, that you will find `senda` useful.

Please direct any questions and feedbacks to
[us](mailto:lars.kjeldgaard@eb.dk)!

If you want to contribute (which we encourage you to), open a
[PR](https://github.com/ebanalyse/senda/pulls).

If you encounter a bug or want to suggest an enhancement, please 
[open an issue](https://github.com/ebanalyse/senda/issues).
