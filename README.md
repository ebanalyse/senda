# senda <img src="logo.png" align="right" height=250/>

![Build status](https://github.com/ebanalyse/senda/workflows/build/badge.svg)
![PyPI](https://img.shields.io/pypi/v/senda.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

`senda` is a small python package for fine-tuning transformers for 
sentiment analysis (and text classification tasks in general).

`senda` builds on the excellent `transformers.Trainer` API (all credit goes to the `Huggingface` team).

## Installation guide
`senda` can be installed from [PyPI](https://pypi.org/project/senda/) with 

```
pip install senda
```

If you want the development version then install directly from [GitHub](https://github.com/ebanalyse/senda).

## How to use

You can use `senda` to fine-tune **any** transformer for **any** text classification task in **any** language.

Here we will go through how to use `senda` for fine-tuning a transformer for detecting the polarity ('positive', 'neutral' or 'negative')
of Danish Tweets. For training we use more than 5,000 Danish Tweets kindly annotated
and hosted by the [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#twitter-sentiment) (thanks!).

First, load Danish Tweets annotated with polarity.

```python
from senda import get_danish_tweets
df_train, df_eval, df_test = get_danish_tweets()
```
Note, that the datasets must be DataFrames containing the columns 'text' and 'label', e.g.

```python
df_train
                                             text    label
Cepos: Vi bør diskutere, hvordan vi afvikler j...  neutral
Avis: FC København og Brøndby IF i duel om Ste...  neutral
@PeterThorup @IntactDenmark Nej - endnu ikke -...  positiv
That was pretty close. Theresa May fortsætter ...  neutral
Så er der ny Facebook-side til min nye forretn...  positiv
                                              ...      ...
@MtnTeit @aeldresagen Helt enig. Vi må bare ik...  negativ
@PrmMortensen @Marchen_Neel @larsloekke @oeste...  negativ
Hvordan sikrer vi ØKONOMISK RENTABLE REGIONALE...  neutral
@JanEJoergensen @24syv @DanskDf1995 @Spolitik ...  negativ
@Fonoudi6eren Ikke enig! Synes vi var godt med...  positiv
```

Next, instantiate and set up the model.

```python
from senda import Model, compute_metrics
from transformers import EarlyStoppingCallback

m = Model(train_dataset = df_train, 
          eval_dataset = df_eval,
          transformer = "Maltehb/danish-bert-botxo",
          labels = ['negativ', 'neutral', 'positiv'],
          tokenize_args = {'padding':True, 'truncation':True, 'max_length':512},
          training_args = {"output_dir":'./results',          
                           "num_train_epochs": 4,             
                           "per_device_train_batch_size":8,   
                           "evaluation_strategy":"steps",
                           "eval_steps":100,
                           "logging_steps":100,
                           "learning_rate":2e-05,
                           "weight_decay": 0.01,
                           "per_device_eval_batch_size":32,   
                           "warmup_steps":100,                
                           "seed":42,
                           "load_best_model_at_end":True,
                           },
           trainer_args = {'compute_metrics': compute_metrics,
                           'callbacks':[EarlyStoppingCallback(early_stopping_patience=4)],
                           }
           )
```

Now, all there is left is to initialize the model (including the `transformers.Trainer`) and train it:

```python
# initialize Trainer
m.init()
# run training
m.train()
```

The model can then be evaluated on the test set:

```python
m.evaluate(df_test)
{'eval_loss': 0.5771588683128357, 'eval_accuracy': 0.7664399092970522, 'eval_f1': 0.7290485787279956, 'eval_runtime': 4.2016, 'eval_samples_per_second': 104.959}
```

Predict new observations:

```python
text = "Sikke en dejlig dag det er i dag"
# in English: 'What a lovely day'
m.predict(text)
PredictionOutput(predictions=array([[-1.2986785 , -0.31318122,  1.2002046 ]], dtype=float32), label_ids=array([0]), metrics={'test_loss': 2.7630457878112793, 'test_accuracy': 0.0, 'test_f1': 0.0, 'test_runtime': 0.07, 'test_samples_per_second': 14.281})

m.predict(text, return_labels=True)
['positiv']
```

## `senda` model available on Huggingface

As you see, the model above achieves an accuracy of 0.77 and a macro-averaged F1-score of 0.73 on a small test data set, that [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#twitter-sentiment) provides.

The model is published on [Huggingface](https://huggingface.co/pin/senda).

Here is how to download and use the model with PyTorch:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
tokenizer = AutoTokenizer.from_pretrained("pin/senda")
model = AutoModelForSequenceClassification.from_pretrained("pin/senda")

# create 'senda' sentiment analysis pipeline 
senda_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

senda_pipeline("Sikke en dejlig dag det er i dag")
[{'label': 'positiv', 'score': 0.7678486704826355}]
```

The model can most certainly be improved, and we encourage all NLP-enthusiasts to train a better model - you can use the `senda` package to do this.

## Background
`senda` is developed as a part of [Ekstra Bladet](https://ekstrabladet.dk/)’s activities on Platform Intelligence in News (PIN). PIN is an industrial research project that is carried out in collaboration between the [Technical University of Denmark](https://www.dtu.dk/), [University of Copenhagen](https://www.ku.dk/) and [Copenhagen Business School](https://www.cbs.dk/) with funding from [Innovation Fund Denmark](https://innovationsfonden.dk/). The project runs from 2020-2023 and develops recommender systems and natural language processing systems geared for news publishing, some of which are open sourced like `senda`.

## Shout-outs
- Thanks to [Alexandra Institute](https://alexandra.dk/) for doing all of the heavy lifting by annotating [Danish tweets]((https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#twitter-sentiment)) (and publishing them).

## Contact
We hope, that you will find `senda` useful.

Please direct any questions and feedbacks to
[us](mailto:lars.kjeldgaard@eb.dk)!

If you want to contribute (which we encourage you to), open a
[PR](https://github.com/ebanalyse/senda/pulls).

If you encounter a bug or want to suggest an enhancement, please 
[open an issue](https://github.com/ebanalyse/senda/issues).
