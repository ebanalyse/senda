
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
from senda import Model, ModelAnalytical
import seaborn as sns
import matplotlib as plt
m = Model(transformer = "pin/analytical")
m.init()
#from SENDA import get_danish_tweets
#train, eval, test = get_danish_tweets()
import pandas as pd

# helper function to convert numpy types to native python types (to avoid app crashing).
def convert_native_python(x):
    return getattr(x, "tolist", lambda: x)()

# function to compute aggregated sentiment scores on article level.
def predict_article(text):

    sents = sent_tokenize(text)
    data = {'text': sents, 'label': [m.labels[0]]*len(sents)}
    data = pd.DataFrame(data)

    preds = m.predict(data).predictions
    
    # compute predicted labels.
    preds = np.argmax(preds, -1)
    
    #### MEAN SENTIMENT SCORE####
    # compute mean sentiment "score" across sentences.
    mean = convert_native_python(preds.mean() - 1)
    
    # compute article label
    if mean > 0.1:
        label_article = "positiv"
    elif mean < -0.15:
        label_article = "negativ"
    else:
        label_article = "neutral"

    # compute header label
    if preds[0] == 0:
        label_header = "negativ"
    elif preds[0] == 1:
        label_header = "neutral"
    else:
        label_header = "positiv"
     
    #### Ratio between positive and negative sentences ####
    count_pos = sum(preds == 2)
    count_neg = sum(preds == 0)
    count_neutral = sum(preds == 1)
    # add 1 to avoid division with zero.
    ratio_pos_neg = (count_pos + 1) / (count_neg + 1)
    ratio_pos_neg = convert_native_python(ratio_pos_neg)

    return {'label_article': [label_article],
            'label_header': [label_header],
            'mean': [mean], 
            'ratio_pos_neg': [ratio_pos_neg],
            'ratio_loaded': [(count_pos+count_neg)/len(sents)],
            'n_positiv': [count_pos],
            'n_negativ': [count_neg],
            'n_neutral': [count_neutral]}

def compute_statics(section = "Sport", n=500):

    if section == "Alle":
        section_select = None
    else:
        section_select = section

    from ebpaperboy.news import get_news
    newspaper = get_news(publication = "ekstrabladet",
                         n_limit = n,
                         order_by = "first_published",
                         section_name = section_select)
    import time

    start = time.time()
    preds = []
    for i, x in enumerate(newspaper.news):
        try:
            x = newspaper.news[i].get_clean()
            res = predict_article(x)
            res = pd.DataFrame.from_dict(res)
            preds.append(res)
        except: 
            pass
    # results = [sent(x) for x in newspaper.news]
    end = time.time()
    print(end - start)
    preds = pd.concat(preds)
    preds.to_csv(f'{section}.csv', index = False)

    # preds = pd.read_csv("....csv")
    out = sns.histplot(data=preds, x="mean", kde = True, bins = 30)
    out.set(title=f'Sektion: {section}',xlabel='Polarity(mean)', ylabel='Count')
    out.get_figure().savefig(f'{section}.png')
    out.get_figure().clf()

    # import numpy as np
    var = preds["mean"]

    from scipy.stats import kurtosis, skew
    import numpy as np
    import math
    #percentiles = np.percentile(var, [0,10,20,30,40,50,60,70,80,90,100]).tolist()
    #percentiles = [round(x, 2) for x in percentiles]
    statistics = {
        "Section": section,
        "Mean": [round(np.mean(var), 2)],
        "Median": [round(np.median(var), 2)],
        "Standard Deviation": [round(np.std(var), 2)],
        "Skewness": [round(skew(var), 2)],
        "Kurtosis": [round(kurtosis(var), 2)],
        #"Deciles": percentiles
    }
    pd.DataFrame.from_dict(statistics, orient = "index")

    sections = ["Erhverv",
                "Danske kendte",
                "Videnskab",
                "Politik",
                "Danske kongelige",
                "Sport",
                "Dansk fodbold",
                "Videnskab",
                "Krimi",
                "Samfund",
                "Krimi",
                "112",
                "Teknologi"]