import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, precision_recall_fscore_support
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from danlp.datasets import TwitterSent
from .angry_tweets import get_angrytweets
from typing import List, Set, Dict, Tuple, Optional, Union
import re

def validate_annotations(x: str) -> str:
    """Validate Angry Tweets Annotations 

    Helper function. If a text is annotated with different labels, 
    replace with 'skip'. 

    Args:
        x (str): Annotation.

    Returns:
        str: Adjusted annotation, single value.
    """
    x = eval(x)
    if len(set(x)) == 2:
        x == ['skip']
    return x[0]

def get_danish_tweets(frac: float = 0.8,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get Danish Tweets

    Gets Danish Tweets annotated kindly annotated and
    hosted by [Alexandra Institute](https://github.com/alexandrainst/danlp/blob/master/docs/docs/datasets.md#twitter-sentiment).
    Remember to set up your Twitter Dev login according
    to instructions for danlp.datasets.TwitterSent() and
    danlp.datasets.AngryTweets() in order to download tweets. 

    Args:
        frac (float, optional): Fraction of data to be sampled
            for training split.
        random_state (int, optional): Fix random state for sampling. 
            Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train,
            eval and test data splits.
    """

    # load 'danlp' TwitterSent tweets.
    test, model_data = TwitterSent().load_with_pandas()
    #model_data = pd.read_csv("twitter_train.csv")
    #test = pd.read_csv("twitter_test.csv")
    model_data = [model_data[['text', 'polarity']].rename(columns={'polarity': 'label'})]
    test = test[['text', 'polarity']].rename(columns={'polarity': 'label'})

    # load 'danlp' AngryTweets tweets.
    angry_tweets = get_angrytweets()
    # angry_tweets = pd.read_csv("angrytweets.csv")
    angry_tweets['label'] = angry_tweets['annotation'].map(validate_annotations)
    angry_tweets = angry_tweets.loc[angry_tweets['label'] != 'skip',]
    angry_tweets = angry_tweets[['text', 'label']]
    model_data.append(angry_tweets)

    # combine data.
    out = pd.concat(model_data)
    
    # make sure index is OK.
    out = out.reset_index()

    # sample data randomly in training and evaluation data sets.
    train = out.sample(frac=frac, random_state=random_state)
    eval = out.drop(train.index)

    return train, eval, test

class TextClassificationDataset(torch.utils.data.Dataset):
    """Torch Text Classification Dataset"""
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(pred) -> dict:
    """Compute Performance Metrics
    
    Computes accuracy and F1 score (macro-averaged).

    Args:
        pred: actual and predicted labels.

    Returns:
        dict: accuracy and f1 score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1': f1
    }

def strip_urls(x: str):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE)

class Model:
    """senda Text Classification Model
    
    A model for Text Classification, e.g. for Sentiment
    Analysis. The model is set up using the `transformers.Trainer`
    API. 
    
    The model can be trained with the `train` method. Afterwards
    new observations can be predicted with `predict` and
    the model can be evaluated with `evaluate`.

    Examples:
        Fine-tune Danish BERT for detecting polarity of Danish Tweets
        >>> from senda import Model, get_danish_tweets
        >>> df_train, df_eval, df_test = get_danish_tweets()
        >>> m = Model(train_dataset = df_train, 
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
        >>> m.init()
        >>> m.train()

    Attributes:
        Trainer: Text Classification Trainer initialized with
            the init() method.
    """

    def __init__(self,
                 transformer: str = "Maltehb/danish-bert-botxo",
                 labels: str = ['negativ', 'neutral', 'positiv'],
                 tokenize_args: dict = {'padding':True, 'truncation':True, 'max_length':512},
                 training_args: dict = {"output_dir":'./results',          # output directory
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
                                  },
                trainer_args: dict = {'compute_metrics': compute_metrics,
                              'callbacks':[EarlyStoppingCallback(early_stopping_patience=4)],
                              },
                train_dataset: pd.DataFrame = None,
                eval_dataset: pd.DataFrame = None,
                strip_urls: bool = True) -> None:
        """Initiate senda Text Classification Model

        Args:
            transformer (str, optional): Pretrained transformer. Defaults to "Maltehb/danish-bert-botxo".
            labels (str, optional): Labels for text classification task. Defaults to ['negativ', 'neutral', 'positiv'].
            tokenize_args (dict, optional): Args for tokenization. Defaults to {'padding':True, 'truncation':True, 'max_length':512}.
            training_args (dict, optional): Training Arguments. Defaults to {"output_dir":'./results',          # output directory "num_train_epochs": 4,              # total # of training epochs "per_device_train_batch_size":8,  # batch size per device during training "evaluation_strategy":"steps", "eval_steps":100, "logging_steps":100, "learning_rate":2e-05, "weight_decay": 0.01, "per_device_eval_batch_size":32,   # batch size for evaluation "warmup_steps":100,                # number of warmup steps for learning rate scheduler "seed":42, "load_best_model_at_end":True, }.
            trainer_args (dict, optional): Optional arguments for Trainer. Defaults to {'compute_metrics': compute_metrics, 'callbacks':[EarlyStoppingCallback(early_stopping_patience=4)], }.
            train_dataset (pd.DataFrame, optional): Training Dataset. The dataset must contain columns 'text' and 'label'. Defaults to None.
            eval_dataset (pd.DataFrame, optional): Evaluation Dataset. The dataset must contain columns 'text' and 'label'. Defaults to None.
            strip_urls (bool, optional): Strip texts for URLs as part of the preprocessing. Defaults to True.
        """
        
        # mapping of labels.
        id2label = {}
        for idx, label in enumerate(labels):
            id2label[idx] = label
        self.id2label = id2label 

        label2id = {}
        for idx, label in enumerate(labels):
            label2id[label] = idx
        self.label2id = label2id  
        self.labels = labels

        # initialize model.
        self.model = AutoModelForSequenceClassification.from_pretrained(transformer, 
                                                                        id2label=id2label, 
                                                                        label2id=label2id,
                                                                        num_labels=len(labels))
        
        self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        self.tokenize_args = tokenize_args
        self.training_args = training_args
        self.trainer_args = trainer_args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.strip_urls = strip_urls 

    def preprocess_data(self, df: pd.DataFrame) -> torch.utils.data.Dataset:
        """Preprocess Dataset

        Args:
            df (pd.DataFrame): Dataset.

        Returns:
            Dataset: torch Dataset after preprocessing. 
        """

        if df is None:
            return None
        
        texts = df['text'].values
        labels = df['label'].values

        if self.strip_urls:
            texts = list(map(strip_urls, texts))            

        # tokenize texts and map labels to ids.
        texts = self.tokenizer(texts, **self.tokenize_args)        
        labels = [self.label2id.get(label) for label in labels]

        return TextClassificationDataset(texts, labels)

    def init(self):
        """Iniatiate model by preparing Trainer"""

        trainer = Trainer(model=self.model,                                       # the instantiated 🤗 Transformers model to be fine-tuned
                          args=TrainingArguments(**self.training_args),           # training arguments, defined above
                          train_dataset=self.preprocess_data(self.train_dataset), # training dataset
                          eval_dataset=self.preprocess_data(self.eval_dataset),   # evaluation dataset
                          **self.trainer_args)

        setattr(self, "Trainer", trainer)

    def evaluate(self, dataset: pd.DataFrame = None, **kwargs) -> dict:
        """Evaluate model with Trainer

        Args:
            dataset (pd.DataFrame, optional): Dataset to evaluate
                model on.

        Returns:
            dict: evaluation results.
        """
        dataset = self.preprocess_data(dataset)
        out = self.Trainer.evaluate(dataset, **kwargs)
        return out

    def train(self):
        """Train model with Trainer."""
        self.Trainer.train()
        
    def predict(self, data: Union[str, List, pd.DataFrame], return_labels=False, **kwargs):
        """Compute Predictions

        Args:
            data (Union[str, List, pd.DataFrame]): New observations to compute
                predictions for. If you provice a DataFrame it must contain
                columns 'text' and 'label'.
            return_labels (bool, optional): Return labels only? Defaults to False.
            kwargs: Optional arguments for Trainer.predict().

        Returns:
            Predictions.
        """

        if isinstance(data, str):
            data = [data]

        if isinstance(data, list):
            data = {'text': data, 'label': self.labels[0]*len(data)}
            data = pd.DataFrame(data)

        dataset = self.preprocess_data(data)
        preds = self.Trainer.predict(dataset, **kwargs)

        # compute predicted labels
        if return_labels:
            preds = np.argmax(preds.predictions, -1)
            preds = [self.id2label.get(id) for id in preds]

        return preds

    def save(self, dir: str):
        """Save model and tokenizer

        Args:
            dir (str): directory to save model in.
        """
        self.Trainer.save_model(dir)
        self.tokenizer.save_pretrained(dir)
        print("Model saved.")


