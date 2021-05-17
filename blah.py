from senda import Model, get_danish_tweets

df_train, df_eval, df_test = get_danish_tweets()

fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

def run_training(frac):
    trn = df_train.sample(frac = frac, random_state = 42)
    training_args = {"output_dir":'./results',          # output directory
                                  "num_train_epochs": 10,              # total # of training epochs
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
                                  }
    m = Model(train_dataset = trn, 
              eval_dataset = df_eval,
              training_args = training_args)
    m.init()
    m.train()
    results = m.evaluate(df_test)
    results = {'n_obs': [len(trn)], 'accuracy': [results.get('eval_accuracy')]}
    return results

out = [run_training(x) for x in fracs]

import pandas as pd

[pd.DataFrame.from_dict(x) for x in out]
out = 

pd.DataFrame.from_dict(out[0])


res = pd.concat() 
results = pd.concat(out)



# senda distribution
from senda import get_danish_analytical_data
trn, test = get_danish_analytical_data()
trn_obj = trn.label.tolist().count("objektivt")
trn_sub = trn.label.tolist().count("subjektivt")
tst_obj = test.label.tolist().count("objektivt")
tst_sub = test.label.tolist().count("subjektivt")
trn_sub/(trn_sub + trn_obj)
tst_sub/(tst_sub + tst_obj)

# sentida
from sentida import Sentida
Sentida().sentida(text,
                  output = ["mean", "total", "by_sentence_mean", "by_sentence_total"],
                  normal = True,
                  speed = ["normal", "fast"]
                  )

# Speed parameter does not have an effect in version <0.2.1

out

# pip install sentida

from sentida import Sentida

sent = Sentida()

def get_sentida_label(text = "Det er så dejligt! Helt fantastisk!"):
    score = sent.sentida(text = text,
                 output = "mean",
                 normal = False,
                 speed = "normal")

    if score < -1.1:
        label = 'negativ'
    elif score <= 1.1:
        label = 'neutral'
    elif score > 1.1:
        label = 'positiv'

    return label
