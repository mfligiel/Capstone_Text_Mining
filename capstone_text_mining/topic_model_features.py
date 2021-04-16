#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:07:14 2021
@author: zjafri
This code creates a function generate a list of topic probabilities using Latent Direchlet Allocation (LDA) for a given list of keywords and number of topics selected. The ouput is an nxp-1 array for n keywords and p number of topics given. Optional parmeters are provided for LDA.
"""

# Define function to clean text
def clean_text(text):
    import re
    import nltk as nltk
    import string as string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    
    nltk.download('stopwords', quiet = True)
    nltk.download('wordnet', quiet = True)
    nltk.download('punkt', quiet = True)
    stop_words = set(stopwords.words('english'))
    
    exclude = set(string.punctuation)
    
    for i in ["'", '“', '”']:
        exclude.add(i)
    
    lemma = WordNetLemmatizer()
    
    spec_char_free = re.sub('[^a-zA-Z0-9 @ . , : - _]', '', text)
    stop_free = ' '.join(word for word in str(spec_char_free).lower().split() if word not in stop_words)
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    tokenized = word_tokenize(normalized)
    return tokenized


# Define function to load keywords
def topic_features(data, num_topics, filter_extremes_no_below = 0, filter_extremes_no_above = 1, output_visual = False, chunksize = 2000, iterations = 50, passes = 3, eval_every = None, random_state = 42):
    
    """
    
    Parameters
    ----------
    data : list object
        A list of keywords
    num_topics : int
        The number of desired topics for the LDA algorithm.
    output_visual: bool
        Generates a visual output for the LDA algorithm Default value is False.
    filter_extremes_no_below : int
        The minimum number of instances for a token within the dictionary to be considered for topic modeling. Default value of 0.
    filter_extermes_no_above : float
        The maximum percentage a token can comprise of the dictionary. Value between 0 and 1. Default value of 1.
    chunksize : int
        Number of documents to be used in each training chunk. Default value of 2000.
    iterations : int
        Maximum number of iterations through the corpus when inferring the topic distribution of a corpus. Default value of 50.
    passes: int
        Number of passes through the corpus during training. Default value of 3.
    eval_every: int
         Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x. Default value of None.
    random_state: int
        Either a randomState object or a seed to generate one. Useful for reproducibility. Default value of 42.
    Returns
    -------
    If output_visual is set to False then an nxp-1 array where n is the number of keywords in the list and p is the number of topics selected. P-1 dimensions are returned as one dimension of p is dropped to eliminate multicolinearity amongst the features. Text cleaning steps are applied to the keywords, including the removal of special characters, stopwords, and lemmatization. If output_visual is set to True, then a pyLDAvis object visualizing the LDA results.
    
    """
    
    # Check if data is a list
    if isinstance(data, list) == False:
        print('Data variable must contain a list of keywords!')   
    else:
        
        from gensim import corpora
        from gensim.models import LdaMulticore
        import multiprocessing
        import numpy as np
#        import pandas as pd

        
        # Create list of clean keywords
        keywords_clean = list(map(clean_text, data))
        
        # Create dictionary
        dictionary = corpora.Dictionary(keywords_clean)
        dictionary.filter_extremes(no_below = filter_extremes_no_below, no_above = filter_extremes_no_above)
        
        # Convert keywords into document term matrix
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in keywords_clean]
        
        # Define number of processes and workers
        num_processors = multiprocessing.cpu_count()
        workers = num_processors - 1  
        
        # Run LDA model
      
        ldamodel = LdaMulticore(corpus = doc_term_matrix, \
                                     id2word = dictionary, \
                                     chunksize = chunksize, \
                                     eta = 'auto', \
                                     num_topics = num_topics, \
                                     iterations = iterations, \
                                     passes = passes, \
                                     eval_every = eval_every, \
                                     workers = workers, \
                                     random_state = random_state)
            
        # Apply LDA Model to document term matrix to assign topic scores
        lda_documents = ldamodel[doc_term_matrix]
        
        # Select highest probability for each document
        doc_max_topic = [max(prob, key = lambda y:y[1]) for prob in lda_documents]
        
        # Select topic number only
        doc_max_topic = [i[0]+1 for i in doc_max_topic]
        
        # Create list of all document topic probabilities
        doc_topic_probs = [prob for prob in lda_documents]
        
        # Create a list of all topic probabilities to unpack nested list and tuples
        prob_list = []
        for i in doc_topic_probs:
                      for j in i:
                        prob_list.append(j[1])
        
        # Convert to array
        probs = np.array(prob_list)
        
        # Reshape
        probs = probs.reshape((len(lda_documents), num_topics))
        
#        # Convert to dataframe
#        df_probs = pd.DataFrame(probs)
#        df_probs = df_probs.iloc[:, 1:]
        
        # Save result
        topic_model_result = probs[:, 1:]
              
        if output_visual == False:
            
            return topic_model_result
        
        else:
                       
            import pyLDAvis.gensim
            import warnings
            
            # Suppress warnings from pyLDAvis
            warnings.filterwarnings('ignore', category = DeprecationWarning)
            
            # Create visualization for LDA Model
            topic_model_visual = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics = False, mds = 'mmds')
            return pyLDAvis.display(topic_model_visual)
