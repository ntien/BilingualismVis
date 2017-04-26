#nltk
#scikit learn
from __future__ import print_function
from __future__ import division

import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

languages = ["french", "spanish", "chinese", "mandarin", "german","russian"]
words = ["family", "bilingual", "biligualism", "multilingualism", "mother"]

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
            print("\n")
            print("Topic #%d:" % topic_idx)
            print(" ".join([feature_names[i]
              for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def get_top_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
      topics[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words -1:-1]]
    return topics

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
        l = re.sub(r'Spanish','',l)
        l = re.sub(r'French','',l)
        l = re.sub(r'Chinese','',l)
        l = re.sub(r'family','',l)
        l = re.sub(r'bilingualism','',l)
        l = re.sub(r'multilingualism','',l)
        l = re.sub(r'bilingual','',l)
        l = re.sub(r'mother','',l)
        ff.append(l)
    files.append(ff)

for i,doc in enumerate(files):
  name = doc[0]
  record[i+1] = name

data = [" ".join(doc) for doc in files]


n_topics = 29
maxdocfreq = 0.63
mindocfreq = 12
n_features = 1000
n_top_words = 20


tf_vectorizer = CountVectorizer(max_df=maxdocfreq, min_df=mindocfreq, max_features=n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(data)


tfidf_vectorizer = TfidfVectorizer(max_df=maxdocfreq, min_df=mindocfreq, max_features=n_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data)

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5)
lda.fit(tfidf)

distrs = lda.transform(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#print_top_words(lda, tfidf_feature_names, n_top_words)
topics_to_words = get_top_words(lda, tfidf_feature_names, n_top_words)

def get_top_topic(sim_matrix):
  toptopics = {}
  for i in range(len(sim_matrix)):  #there should be 30
    toptopics[i] = sim_matrix[i].argmax()
  return toptopics
print_top_words(lda, tfidf_feature_names, n_top_words)

tt = get_top_topic(distrs)
#pairwise_similarity = distrs * distrs.transpose
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
s = sparse.csr_matrix(distrs)

sims = cosine_similarity(s)


'''
top = get_top_topic(distrs)
for doc in top:
  print("Document: " + str(doc))
  print(topics_to_words[top[doc]])
  print("\n")
'''




