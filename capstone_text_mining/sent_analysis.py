#standard imports
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

#function
def sent_analysis(data, keywordvar):
    """
    

    Parameters
    ----------
    data : DataFrame
        A Dataframe that is already cleaned of nulls and ready to model
    keywordvar : string
        The name of the variable which has the keyword  

    Returns
    -------
    The dataset with associated scores.

    """
    #check that the data doesn't have columns with our names:
    assert 'pos' not in data.columns
    assert 'neg' not in data.columns
    assert 'neu' not in data.columns
    assert 'composite' not in data.columns
    #run
    analyzer = SentimentIntensityAnalyzer()
    #reduce the data by only runnin on distrinct keywords
    df_small = pd.DataFrame(data[keywordvar])
    df_small.drop_duplicates(inplace=True)
    df_small.reset_index(inplace=True, drop=True)
    df_small['scores'] = df_small[keywordvar].apply(lambda x: analyzer.polarity_scores(str(x)))
    #take the scores out of the dictionary and make a dataframe
    print(df_small)
    sent_attributes = pd.DataFrame.from_dict(df_small['scores'].to_list())
    df_scores =  df_small.merge(sent_attributes, left_index=True, right_index=True)
    df_scores.drop('scores', axis=1, inplace=True)
    print(df_scores)
    #join back in and return
    ret = data.merge(df_scores, on=keywordvar)
    return ret
