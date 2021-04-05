#Spacy Helper Functions for POS, dependencies, entity label types respectively
def nlp_xtrct_pos(string, nlp):
  vv = nlp(str(string))
  l_ret = []
  for t in vv:
    l_ret.append(t.pos_)
  return l_ret

def nlp_xtrct_dep(string, nlp):
  vv = nlp(str(string))
  l_ret = []
  for t in vv:
    l_ret.append(t.dep_)
  return l_ret

def nlp_xtrct_elab(string, nlp):
  vv = nlp(str(string))
  l_ret = []
  for t in vv.ents:
    l_ret.append(t.label_)
  return l_ret

def spacyanalysis(data, keyword, ner=True, pos=True, syndeps=True):
    """
    

    Parameters
    ----------
    data : dataframe
        The data to use
    keyword : string
        Name of the keyword column to mine.
    ner : Bool, optional
        Whether or not to run named entity recognition. The default is True.
    pos : Bool, optional
        Whether or not to run the part of speech analysis. The default is True.
    syndeps : TYPE, optional
        Whether or not to run the syntactid dependency analysis. The default is True.

    Returns
    -------
    A merged back dataframe with the new features.

    """
    #Load in the spacy language model
    nlp = spacy.load("en_core_web_sm")
    #look at individual phrases
    df_small = pd.DataFrame(data[keyword])
    df_small.drop_duplicates(inplace=True)
    #make multilabel binarizer
    mlb = MultiLabelBinarizer()
    #apply each of the possible choices:
    if ner == True:
        print('running ner')
        df_small['ner'] = df_small['KEYWORD'].apply(nlp_xtrct_elab, nlp=nlp)
        #add in binarized
        el =\
        pd.DataFrame(mlb.fit_transform(df_small['ner']),columns=mlb.classes_ +'ent_labels', index=df_small.index)
        df_small = df_small.merge(el, left_index=True, right_index=True, how='left' )
        df_small.drop('ner', axis=1, inplace=True)
    if pos == True:
        print('running pos')
        df_small['pos'] = df_small['KEYWORD'].apply(nlp_xtrct_pos, nlp=nlp)
        #add in binarized
        psp =\
        pd.DataFrame(mlb.fit_transform(df_small['pos']),columns=mlb.classes_ +'partspeech', index=df_small.index)
        df_small = df_small.merge(psp, left_index=True, right_index=True, how='left' )
        df_small.drop('pos', axis=1, inplace=True)
    if syndeps == True:
        print('running syntactic dependencies')
        df_small['syndeps'] = df_small['KEYWORD'].apply(nlp_xtrct_dep, nlp=nlp)
        #add in binarized
        dep =\
        pd.DataFrame(mlb.fit_transform(df_small['syndeps']),columns=mlb.classes_ +'dependencies', index=df_small.index)
        df_small = df_small.merge(dep, left_index=True, right_index=True, how='left' )
        df_small.drop('syndeps', axis=1, inplace=True)
    #finally, merge back
    data = data.merge(df_small, left_on='KEYWORD', right_on='KEYWORD', how='left')
    return data
