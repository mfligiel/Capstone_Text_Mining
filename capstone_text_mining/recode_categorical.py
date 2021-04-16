#standard imports
import pandas as pd
import numpy as np

def recode_categorical(data, keywordvar, n_variables_max=5, perc_count = 0.05):
    """
    

    Parameters
    ----------
    data: DataFrame
        A Dataframe that is already cleaned of nulls and ready to model
    keywordvar : string
        The name of the variable which has the keyword  
    n_variables_max : int
        The number of maxiumum variables from each of the individual categorical variables.  Any more than this get grouped into an other.
    perc_count : float
        A threshhold for percentage of rows with a result...to be further developed, not functional yet.

    Returns
    -------
    The dataset with cleaned categorical variables

    """ 
    cmn_lst = data.drop(keywordvar, axis=1).select_dtypes(exclude=['int', 'float']).columns.to_list()
    print(cmn_lst)
    for cmn in cmn_lst:
      if data[cmn].nunique() == 1:
        print('dropping ' + str(cmn) + 'as there is only one unique value.')
        data.drop(cmn, axis=1, inplace=True)
        cmn_lst.remove(cmn)
      else:
        vals_keep = data[cmn].value_counts().head(n_variables_max).index
        data.loc[~data[cmn].isin(vals_keep), cmn] = "Other/Missing"
    data = pd.get_dummies(data, columns = cmn_lst)
    return data

