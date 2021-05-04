#standard imports
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

def model(data, y_var, categorical = False, grid_search = False,\
          keywordvar = None, model_args = None):
    """
    

    Parameters
    ----------
    data : DataFrame
        A Dataframe that is already cleaned of nulls and ready to model
    y_var : Float, Int
        A y variable.  
    categorical : TYPE, optional
        DESCRIPTION. The default is False.
    grid_search : TYPE, optional
        DESCRIPTION. The default is False.
    keywordvar :  STRING, optional
        The keyword variable, can be dropped.
    model_args :  DICT, optional
        Arguments for the model.

    Returns
    -------
    rf : a fitted random forest.

    """
    if keywordvar:
      data = data.drop(keywordvar, axis=1)
    if categorical == True:
        data[y_var] = (df[y_var] > 0).astype(int)
        #from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(\
            data.drop(y_var, axis=1), data[y_var], test_size=0.2, random_state=42)
        if grid_search == True:
            param_grid = dict(class_weight = ['balanced', None],\
                          min_samples_leaf=[0.01, 0.03, 0.05, 5],\
                  max_features=['auto', None],\
                      max_depth = [5, 10, 15],\
                      ccp_alpha = [0.001, 0])
            rf = RandomForestClassifier()
            grid_searchrf = GridSearchCV(rf, param_grid, scoring = 'balanced_accuracy', \
                      refit='balanced_accuracy', verbose=0)
            grid_searchrf.fit(X_train, y_train.astype(int))
        else:
          if model_args != None:
            rf = RandomForestClassifier(**model_args)
          else:
            rf = RandomForestClassifier() #add parameters later
            rf.fit(X_train, y_train)
    else:
        X_train, X_test, y_train, y_test = train_test_split(\
            data.drop(y_var, axis=1), data[y_var], test_size=0.2, random_state=42)
        if grid_search == True:
            param_grid = dict( min_samples_leaf=[0.01, 0.03, 0.05, 5],\
                  max_features=['auto', None],\
                      max_depth = [5, 10, 15],\
                      ccp_alpha = [0.001, 0])
            rf = RandomForestRegressor()
            grid_searchrf = GridSearchCV(rf, param_grid, scoring = 'r2', \
                      refit='r2', verbose=0)
            grid_searchrf.fit(X_train, y_train.astype(int))
        else:
          if model_args != None:
            rf = RandomForestRegressor(**model_args)
          else:
            rf = RandomForestRegressor() #add parameters later
            rf.fit(X_train, y_train) 
    return rf
