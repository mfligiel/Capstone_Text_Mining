#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:35:11 2021

@author: zjafri
"""

def hand_label(data, label):

    """
    
    Parameters
    ----------
    data : list object
        A list of keywords
    
    
    Returns
    -------
    An list of 1s and 0s with 1 indicating the presence of the desired keyword token.
    
    """
    
    # Assert data is a list of tokens
    assert isinstance(data, list) == True, 'Data must be a list of string tokens!'  
    
    # Import required snorkel package 
    from snorkel.labeling import labeling_function

    # Define labeling function
    @labeling_function()
    def lf_contains(keyword):
        # Return label of 1 if i in keyword, otherwise 0
        return 1 if str(label).lower() in str(keyword).lower() else 0
       
    return list(map(lf_contains, data))
              
