#standard imports
import pandas as pd
import numpy as np
import seaborn as sns
import re
import csv #used for the embeddings
from tqdm import tqdm #for seeing loop progress
import multiprocessing # used to optimize lda multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
tqdm.pandas()

#these need to be whittled down

#version of this function

def data_clean(dataframe, y, keywordvar, percdrop=.5, ignore=[], drop = [], filldict=None):
    """
    

    Parameters
    ----------
    dataframe : DataFrame
        A Dataframe that is already cleaned of nulls and ready to model
    y : string
        The name of the variable that is outcome
    keywordvar : string
        The name of the variable which has the keyword  
    percdrop : float
        Between 0 and 1; anything above this percentage null is dropped
    ignore : list
        A list of column names to ignore.
    drop : list
        A list of variables to drop.
    filldict : dictionary
        A dictionary that has variable names and what to fillna with

    Returns
    -------
    The dataset ready to model, except that it still has the keyword.
    


    """
    #make a copy
    data = dataframe.copy()
    #first, see if there are any cases where y is null and drop them
    data.dropna(subset=[y], inplace=True) 
    #fill na if dictionary is passed in
    if filldict:
        for cmn in filldict.keys():
            data[cmn].fillna(filldict[cmn], inplace=True)
    #next going through all columns
    for drp in drop:
        try:
          data.drop(drp, axis=1, inplace=True)
        except:
          print('column ' +drp+' not found.')
    for cmn in data.drop(ignore + [y, keywordvar], axis=1).columns:
        percnull = sum(data[cmn].isna())/len(data[cmn]) > percdrop
        if percnull > percdrop:
            print('Dropping column ' + str(cmn) + ', as percent null is: ' + str(percnull*100))
            data.drop(cmn, axis=1, inplace=True)
        else:
            data.dropna(subset=[cmn], inplace=True) 
    return data
