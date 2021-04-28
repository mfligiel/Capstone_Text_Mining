# CAPSTONE-TEXT-MINING README
Version 0.0.9

## What is it?
The Capstone-Text-Mining is a Python package developed for a graduate analytics program capstone project for a specific client. The client is a data analytics consulting and software-as-a-service provider with its own Machine Learning Operations (MLOps) Platform. The client specializes in paid search and marketing analytics. 

The package allows an end user to apply text mining and natural language processing (NLP) techniques to analyze, evaluate, and identify superior keywords for paid search campaigns. It integrates typical text data cleanup steps, text mining and NLP approaches, and modeling techniques to evaluate the effectiveness of the keywords. 


## Main Features
The package provides the following capabilities: 
- Ability to load paid search campaign data, including keywords, campaign metadata, and outcome metrics (e.g., clickthrough rates) 
- Application of common text data cleaning techniques, including removal of punctuation and stopwords, tokenization, etc. 
- Application of various text mining and NLP techniques to the keywords to develop a variety of features. These methods include: 
- Topic Modeling 
- Named Entity Recognition 
- Hand Labeling of text features 
- Graph model of text 
- Sentiment Analysis 
- Regression and classification model creation using a baseline model (without text mining) and with text mining to estimate the improvement or “lift” the keyword provides in terms an outcome metric (e.g., CTR) 
- Data visualization to evaluate the most impactful text-based features to aid in the analysis and evaluation of keywords. This includes graph and SHAP. 


## Where to get it?

The source code is currently hosted on GitHub at: https://github.com/mfligiel/Capstone_Text_Mining 

The package can installed from the Python Package Index (PyPI):

```pip install Capstone_Text_Mining```


## Getting Started
We recommend sourcing or creating a campaign-level paid search dataset. Each record in the data set should represent a specific campaign and keyword option. Additional level of granularity may be added – e.g., by week, day, channel, etc. In addition to the keyword, each record should have some additional campaign variables to establish a baseline for evaluating the effectiveness of the keyword. The various text mining feature engineering functions may then be applied to the keyword(s) to generate various text-based features for your data set. This data can then be visualized. Models can also be applied to the data to evaluate the effectiveness of the keyword selected. 
