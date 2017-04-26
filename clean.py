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


def get_top_topic(sim_matrix):
  toptopics = {}
  for i in range(len(sim_matrix)):  #there should be 30
    toptopics[i] = sim_matrix[i].argmax()
  return toptopics

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

def grid_search():
  n_topics = 10
  maxdocfreq = 0.70
  mindocfreq = 15
  n_features = 1000
  n_top_words = 20

  numdistincttoptopics = 0
  vals = []

  for i in range(60,80):
    for j in range(10,15):
      for k in range(20, 30):
        n_topics = k
        maxdocfreq = i/100
        mindocfreq = j

        tfidf_vectorizer = TfidfVectorizer(max_df=maxdocfreq, min_df=mindocfreq, max_features=n_features, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(data)

        lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5)
        lda.fit(tfidf)

        tf_feature_names = tfidf_vectorizer.get_feature_names()
        distrs = lda.transform(tfidf)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        tt = get_top_topic(distrs)
        x = len(set(tt.values()))
        if x > numdistincttoptopics:
          numdistincttoptopics = x
          vals = [n_topics, maxdocfreq, mindocfreq]

  print(numdistincttoptopics)
  print("\n\n")
  print("{} topics, {} max document frequency, {} min document frequency".format(vals[0],vals[1],vals[2]))


grid_search()
