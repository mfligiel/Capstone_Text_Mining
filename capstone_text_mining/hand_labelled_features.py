#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:35:11 2021

@author: zjafri
"""
# Import packages
from snorkel.labeling import labeling_function
import pandas as pd

def hand_label(data, keyword, label, label_name):

    """
    
    Parameters
    ----------
    data : list object
        A list of keywords
    
    
    Returns
    -------
    An list of 1s and 0s with 1 indicating the presence of the desired keyword token.
    
    """
    
    # Confirm data is in a pandas dataframe and column is a column in the dataframe
    assert isinstance(data, pd.core.frame.DataFrame) == True, 'Data must be a pandas dataframe!'          
    assert isinstance(data[keyword], pd.core.frame.Series) == True, 'Column must be a pandas series!'    
    
    # Define labeling function
    @labeling_function()
    def lf_contains(keyword):
        # Return label of 1 if i in keyword, otherwise 0
        return 1 if str(label).lower() in str(keyword).lower() else 0
       
    data[label_name] = data[keyword].apply(lambda x: lf_contains(str(x)))
