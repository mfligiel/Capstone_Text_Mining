def feature_imp(modelname, count, color="blue"):
    """
    Parameters
    ----------
    model : string
        A model with parameters already selected and fit
    modeltype : string
        Identification of pretermined models.  Need to narrow and discuss.
    color : string
        Color selection for the graph 
    count : int
        The number of features to include in the visualization                                      `                               

    Returns
    -------
    A visualization of the top features in a model analysis    


    """
    modeltype = type(modelname).__name__
    #first, identify the type of model for gathering the features
    if modeltype in ('RandomForestRegressor', 'XGBRegressor'):
      importances = pd.Series(modelname.feature_importances_,index = X_train.columns)
    elif modeltype in ('LogisticRegressor',  'MultinomialNB'):
      importances = pd.Series(mnbmodel.coef_.ravel(),index = X_train.columns)
    #consider how to handle the errors here.  if we limit model selection above, this error message may not be needed.
    else:
      print ("This function only works with certain model types.  Please see documentation.")
      return
    # Sorting the importances
    sorted_importances = importances.sort_values(ascending=False)[0:count]
    # Make a horizontal bar plot
    sorted_importances.plot(kind='barh', color=color, title='Feature Importance')
    plt.show()   