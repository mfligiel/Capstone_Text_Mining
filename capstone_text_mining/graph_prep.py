def graphprep(dataframe, keywordcolumn, metriccolumn, count):

# Create initial dataframe
  graphdf = pd.DataFrame(dataframe, columns = [keywordcolumn, metriccolumn])
  graphdf.columns=['KEYWORD', 'METRIC']
# Clean tokens - remove stop words and lemmatize

# Define parameters to clean text
  stop_words = set(stopwords.words('english'))

  exclude = set(string.punctuation)

  for i in ["'", '“', '”']:
    exclude.add(i)

  lemma = WordNetLemmatizer()

# Define function to clean text
  def clean_text(text):
    spec_char_free = re.sub('[^a-zA-Z0-9 @ . , : - _]', '', text)
    stop_free = ' '.join(word for word in str(spec_char_free).lower().split() if word not in stop_words)
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(lemma.lemmatize(word) for word in punc_free.split())
    tokenized = word_tokenize(normalized)
    return tokenized
# Apply cleaning function to tokens
  graphdf['kw_tokens_clean'] = graphdf.KEYWORD.apply(lambda x: clean_text(str(x)))
# Create list of all clean tokens
  clean_tokens = [y for x in graphdf.kw_tokens_clean for y in x]
# Creat distribution of tokens
  word_dist = nltk.FreqDist(clean_tokens)
  top = word_dist.most_common(count)
# Look at top tokens
  top = [x[0] for x in top]
#Keyword Set-up For Graph
  import itertools
  combol = itertools.combinations(top, 2)
  combol = list(combol)
  df_gr=pd.DataFrame(graphdf, columns=['METRIC', 'kw_tokens_clean'])
  df_grl = df_gr.values.tolist()
  import tqdm
  retlist = []
  for i, j in tqdm.tqdm(combol):
  #pstart=[i,j]
    toret = [x[0] for x in df_grl if i in x[1] and j in x[1]]
    retlist.append([i, j, toret])

  forgraph = pd.DataFrame(retlist)
  forgraph.columns = ['word1', 'word2', 'vals']
  #Make count and mean, and only include cases where the count is greater than 0, implying that they co-appear
  forgraph['num'] = forgraph['vals'].apply(len)
  forgraph['nanmean'] = forgraph.loc[forgraph['num'] > 0, 'vals'].apply(np.nanmean)
  forgraph = forgraph.loc[forgraph['num'] > 0, ['word1', 'word2', 'num', 'nanmean']]
  import networkx as nx
  graph = nx.convert_matrix.from_pandas_edgelist(forgraph, source='word1', target='word2', edge_attr=['num', 'nanmean'])
  # Add attribute for degrees to all nodes
  from collections import defaultdict
  node_dictionary = defaultdict(int)

  for n in graph.nodes():
    node_dictionary[n] = nx.degree(graph, n)

  nx.set_node_attributes(graph, node_dictionary, 'degrees')
  return graph