#standard imports
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

def lift(data,  todrop, y_var, othervars, test_data=None, model_args=None, top_perc = .2, fn = lambda x: x , full_model=None):
    """
    

    Parameters
    ----------
    data : DataFrame
        A Dataframe that is already cleaned of nulls and ready to model
    test_data : DataFrame, optional
        Optional test data for calculating lift on.  If this is included train data is only used to train.
    y_var : Float, Int
        A y variable.  
    othervars : List
        A list of variables which are from text mining; we see the lift of the data with these vs the data without.
    model_args : Dict, optional
        model arguments to pass into the model.
    todrop :  List
        A list of variables to drop from both models.
    top_perc : float
        Default .2, Between 0 and 1.  The proportion of top results used to compare, as this will do regression.
    function : function
        Optional function to apply.
    full_model:
        A full model that can be passed it, so it doesn't need to be retrained

    Returns
    -------
    None.  It does however print information about lift.

    """
    assert top_perc <= 1
    assert top_perc > 0
    datafr = data.copy()
    datafr.drop(todrop, inplace=True, axis=1)
    #print(data.head())
    ### FULL MODEL
    if full_model is not None:
      rf_full = full_model
    else:
      if model_args is not None:
        rf_full = RandomForestRegressor(**model_args, verbose=1)
        print('fitting full model...')
        rf_full.fit(X=datafr.drop(y_var, axis=1), y=datafr[y_var])
      else:
        rf_full = RandomForestRegressor(verbose=1)
        print('fitting full model...')
        rf_full.fit(X=datafr.drop(y_var, axis=1), y=datafr[y_var])
    if test_data:
      test_datafr = test_data.copy()
      test_datafr.drop(todrop, inplace=True, axis=1)
      y_pred_full = rf_full.predict(test_datafr.drop(y_var, axis=1))
      y_act = test_data[y_var]
    else:
      y_pred_full = rf_full.predict(datafr.drop(y_var, axis=1))
      y_act = datafr[y_var]
    #top Xth percentile predicted actual:
    outpt_full = pd.DataFrame(np.vstack([y_pred_full, y_act]).T, columns = ['pred', 'act'])
    #get the 'line' for which we are in the top_perc percent based on predictive
    quant = 1 - top_perc
    val_line_full = outpt_full.pred.quantile(quant)
    #get the sum of the ACTUAL values above that line.
    #NOTE I HAVE LEFT IN THE np.exp - i think I need to make this a parameter, for transformations.
    sum_full = sum(outpt_full.loc[outpt_full['pred'] > val_line_full, 'act'].apply(fn))

    ### BASE MODEL
    if model_args:
      rf_base = RandomForestRegressor(**model_args, verbose=1)
    else:
      rf_base = RandomForestRegressor(verbose=1)
    print('fitting base...')
    rf_base.fit(X=datafr.drop(y_var, axis=1).drop(othervars, axis=1), y=data[y_var])
    if test_data:
      y_pred_base = rf_base.predict(test_datafr.drop(y_var, axis=1).drop(othervars, axis=1))
    else:
      y_pred_base = rf_base.predict(datafr.drop(y_var, axis=1).drop(othervars, axis=1))
    #top Xth percentile predicted actual:
    outpt_base = pd.DataFrame(np.vstack([y_pred_base, y_act]).T, columns = ['pred', 'act'])
    #get the 'line' for which we are in the top_perc percent based on predictive
    val_line_base = outpt_base.pred.quantile(quant)
    #get the sum of the ACTUAL values above that line.
    #NOTE I HAVE LEFT IN THE np.exp - i think I need to make this a parameter, for transformations.
    sum_base = sum(outpt_base.loc[outpt_base['pred'] > val_line_base, 'act'].apply(fn))

    #Calc Lift
    print("Full Model Outcome Count:", str(sum_full))
    print("Base Model Outcome Count:", str(sum_base))
    print("The final lift is:", sum_full/sum_base*100 - 100,  "%")


