#Shap Function 

def shap_visuals(modelname, sample, keyrow):
    """
    Parameters
    ----------
    model : string
        A model with parameters already selected and fit
    modeltype : string
        Identificationof pretermined models 
    sample : int
        The number of values to use in the random sample from X_test
    keyrow: inf
        The row value for additional visualization

    Returns
    -------
    A visualization of the top features in a model analysis    


    """
    import shap
    #first, create the random sample set
    if sample >1000:
      print ("It is required to use a sample size of 1,000 or less")
      return

    elif sample > len(X_test):
          sample=len(X_test)
          print ("Sample size was great than length of test data. Visualization will use all test data.")
    else: 
        Random_X_test=X_test.sample(sample)

    #second, identify the type of model for gathering the features
    modeltype = type(modelname).__name__
    #if random forest, use the fast random forest model
    if modeltype in 'RandomForestRegressor', 'XGBRegressor', 'GradientBoostingRegressor', 'RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier'):
       print ("Running explainer.  This takes awhile.  Be patient.")
       explainer =  shap.TreeExplainer(modelname)
    elif modeltype in ('LogisticRegression', "LinearRegression"):
        print ("Running explainer.  This takes awhile.  Be patient.")
        explainer = shap.KernelExplainer(modelname)
    #consider how to handle the errors here.  if we limit model selection above, this error message may not be needed.
    else: 
      print ("Please choose one of the models listed in the documentation.")
      return
    # Creating Shap Values
    shap_values = explainer.shap_values(Random_X_test)
    # Make standard Shap plot
    shap.summary_plot(shap_values, Random_X_test)
