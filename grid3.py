from __future__ import print_function
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

def get_top_topic(sim_matrix):
  toptopics = {}
  for i in range(len(sim_matrix)):  #there should be 30
    toptopics[i] = sim_matrix[i].argmax()
  return toptopics

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

def grid_search(maxlower, maxupper, topiclower, topicupper, minlower, minupper):
  n_topics = 10
  maxdocfreq = 0.70
  mindocfreq = 15
  n_features = 1000
  n_top_words = 20

  score = 0
  vals = []

  for i in range(maxlower, maxupper):
    for j in range(minlower,minupper):
      for k in range(topiclower, topicupper):
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


        distrs = lda.transform(tfidf)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()

        tt = get_top_topic(distrs)
        distincttoptopics = len(set(tt))
        distinctwords = get_top_words(lda, tf_feature_names, n_top_words)[0]

        thisscore = 0.4*distincttoptopics + 0.6*distinctwords

        if thisscore > score:
          score = thisscore
          vals = [n_topics, maxdocfreq, mindocfreq]


  print(score)
  print(vals)
  print("\n")

#def grid_search(maxlower, maxupper, topiclower, topicupper, minlower, minupper):
'''
grid_search(50,60, 7, 13, 7, 10)
grid_search(50,60, 7, 13, 10, 15)
grid_search(50,60, 13, 20, 7, 10)
grid_search(50,60, 13, 20, 10, 15)
grid_search(50,60, 20, 25, 7, 10)
grid_search(50,60, 20, 25, 10, 15)
grid_search(50,60, 25, 30, 7, 10)
grid_search(50,60, 25, 30, 10, 15)
grid_search(60,70, 7, 13, 7, 10)
grid_search(60,70, 7, 13, 10, 15)
grid_search(60,70, 13, 20, 7, 10)
grid_search(60,70, 13, 20, 10, 15)
grid_search(60,70, 20, 25, 7, 10)
grid_search(60,70, 20, 25, 10, 15)
grid_search(60,70, 25, 30, 7, 10)
grid_search(60,70, 25, 30, 10, 15)

'''
grid_search(70,85, 7, 13, 7, 10)
grid_search(70,85, 7, 13, 10, 15)
grid_search(70,85, 7, 13, 15, 20)
grid_search(70,85, 13, 20, 7, 10)
grid_search(70,85, 13, 20, 10, 15)
grid_search(70,85, 13, 20, 15, 20)
grid_search(70,85, 20, 25, 7, 10)
grid_search(70,85, 20, 25, 10, 15)
grid_search(70,85, 20, 25, 15, 20)
grid_search(70,85, 25, 30, 7, 10)
grid_search(70,85, 25, 30, 10, 15)
grid_search(70,85, 25, 30, 15,20)
'''
grid_search(85,95, 7, 13, 7, 10)
grid_search(85,95, 7, 13, 10, 15)
grid_search(85,95, 7, 13, 15, 20)
grid_search(85,95, 13, 20, 7, 10)
grid_search(85,95, 13, 20, 10, 15)
grid_search(85,95, 13, 20, 15, 20)
grid_search(85,95, 20, 25, 7, 10)
grid_search(85,95, 20, 25, 10, 15)
grid_search(85,95, 20, 25, 15, 20)
grid_search(85,95, 25, 30, 7, 10)
grid_search(85,95, 25, 30, 10, 15)
grid_search(85,95, 25, 30, 15,20)

'''
