#nltk
#scikit learn
from __future__ import print_function
from __future__ import division

import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
            print("\n")
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
              for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def get_top_words(model, feature_names, n_top_words):
    words = []
    for topic_idx, topic in enumerate(model.components_):
      words.extend([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    return len(set(words)), words

files = []
record = {}
for i in range(1,31):
  fname = "interviews/" + str(i) + ".txt"
  with open(fname, "r") as f:
    ff = []
    for line in f:
      if line != "\n" and "Introduction" not in line:
        l = line.split("\n")[0]
        l = re.sub(r'\([0-9]*\)','',l)
        ff.append(l)
    files.append(ff)

for i,doc in enumerate(files):
  name = doc[0]
  record[i+1] = name

data = [" ".join(doc) for doc in files]
'''
n_topics = 10
maxdocfreq = 0.70
mindocfreq = 15
n_features = 1000
n_top_words = 20


print("performing stopping/term frequency analysis")
tf_vectorizer = CountVectorizer(max_df=maxdocfreq, min_df=mindocfreq, max_features=n_features, stop_words='english')

tf = tf_vectorizer.fit_transform(data)
'''

def grid_search():
  n_topics = 10
  maxdocfreq = 0.70
  mindocfreq = 15
  n_features = 1000
  n_top_words = 20

  numdistinctwords = 0
  vals = []
  words = []

  '''
  for i in range(50,96):
    for j in range(2,20):
      for k in range(5, 31):
        '''

  for i in range(75,90):
    for j in range(10,15):
      for k in range(17, 30):
        n_topics = k
        maxdocfreq = i/100
        mindocfreq = j

        tfidf_vectorizer = TfidfVectorizer(max_df=maxdocfreq, min_df=mindocfreq, max_features=n_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)

        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5)
        lda.fit(tfidf)

        tf_feature_names = tfidf_vectorizer.get_feature_names()
        result = get_top_words(lda, tf_feature_names, n_top_words)
        x = result[0]
        words = result[1]
        if x > numdistinctwords:
          numdistinctwords = x
          vals = [n_topics, maxdocfreq, mindocfreq]
          words = words
  print(words)
  print("\n\n")
  print(vals)


grid_search()
