#nltk
#scikit learn
from __future__ import print_function
from __future__ import division
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords as sw
import re

languages = [{"name": "Farsi",
             "children": [
              {"name": "1", "size": 2000}
            ]},
            {"name": "Arabic",
             "children": [
              {"name": "1", "size": 2000}
            ]},
            {"name": "Russian",
             "children": [
              {"name": "2", "size": 2000}
            ]},
            {"name": "Italian",
             "children": [
              {"name": "3", "size": 2000},
              {"name": "30", "size": 2000}
            ]},
            {"name": "Hindi",
             "children": [
              {"name": "4", "size": 2000}
            ]},
            {"name": "Spanish",
             "children": [
              {"name": "5", "size": 2000},
              {"name": "7", "size": 2000},
              {"name": "8", "size": 2000},
              {"name": "10", "size": 2000},
              {"name": "13", "size": 2000},
              {"name": "15", "size": 2000},
              {"name": "17", "size": 2000},
              {"name": "21", "size": 2000},
              {"name": "23", "size": 2000},
              {"name": "24", "size": 2000},
              {"name": "25", "size": 2000}
            ]},
            {"name": "Albanian",
             "children": [
              {"name": "6", "size": 2000}
            ]},
           {"name": "French",
             "children": [
              {"name": "7", "size": 2000},
              {"name": "8", "size": 2000},
              {"name": "10", "size": 2000},
              {"name": "11", "size": 2000},
              {"name": "22", "size": 2000},
              {"name": "28", "size": 2000},
              {"name": "30", "size": 2000}
            ]},
           {"name": "Haitian Creole",
             "children": [
              {"name": "7", "size": 2000},
              {"name": "28", "size": 2000},
            ]},
            {"name": "Korean",
             "children": [
              {"name": "9", "size": 2000},
              {"name": "24", "size": 2000},
              {"name": "26", "size": 2000},
            ]},
            {"name": "Mandarin",
             "children": [
              {"name": "9", "size": 2000},
              {"name": "12", "size": 2000},
              {"name": "14", "size": 2000},
              {"name": "16", "size": 2000},
              {"name": "18", "size": 2000},
              {"name": "19", "size": 2000},
              {"name": "20", "size": 2000},
              {"name": "22", "size": 2000},
              {"name": "29", "size": 2000},
            ]},
            {"name": "Portuguese",
             "children": [
              {"name": "10", "size": 2000},
              {"name": "17", "size": 2000}
            ]},
            {"name": "Thai",
             "children": [
              {"name": "11", "size": 2000}
            ]},
            {"name": "Dutch",
             "children": [
              {"name": "12", "size": 2000}
            ]},
            {"name": "German",
             "children": [
              {"name": "12", "size": 2000}
            ]},
            {"name": "Kashmiri",
             "children": [
              {"name": "12", "size": 2000}
            ]},
            {"name": "Catalan",
             "children": [
              {"name": "13", "size": 2000}
            ]},
            {"name": "Esperanto",
             "children": [
              {"name": "14", "size": 2000}
            ]},
            {"name": "Singlish",
             "children": [
              {"name": "18", "size": 2000}
            ]},
            {"name": "Cantonese",
             "children": [
              {"name": "19", "size": 2000},
              {"name": "22", "size": 2000}
            ]},
            {"name": "Hunan",
             "children": [
              {"name": "19", "size": 2000},
            ]},
            {"name": "Hungarian",
             "children": [
              {"name": "27", "size": 2000},
            ]}
      ]

words = ["family", "bilingual", "biligualism", "multilingualism", "lillehaugen","lille", "page"]
languages = languages[1:]

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
        l = re.sub(r'Spanish','',l)
        l = re.sub(r'French','',l)
        l = re.sub(r'Italian','',l)
        l = re.sub(r'Mandarin','',l)
        l = re.sub(r'Haverford','',l)
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
  title = doc[1].split()
  record[i+1] = " ".join(title)

stop = set(sw.words('english'))
data = [[word for word in doc if word not in stop] for doc in files]
data = [" ".join(doc) for doc in files]
print(data[0])
for d in languages:
  for child in d["children"]:
    x = int(float(child["name"]))
    child["name"] = str(x) + ". " + record[x]



n_topics = 12
maxdocfreq = 0.60
mindocfreq = 7
n_features = 1001
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

#for each topic, grab top 5 documents that have highest distributions of that topic
revdistrs = distrs.T
circles = {}
circles["name"] = "vis"
circles["children"] = []
for i in range(len(revdistrs)):
    docs = [j for j in range(len(revdistrs[i])) if revdistrs[i][j] > 0.3]
    if len(docs) > 0:
      subcircle = {"name": "topic " + str(i)}
      subcircle["children"] = [{"name": str(k) + ". " + record[k+1], "size": 2000} for k in docs]
      circles["children"].append(subcircle)

test = [topic.values() for topic in circles["children"]]
t1 = [item for sublist in test for item in sublist if "topic" not in item]
t2 = [item["name"] for sublist in t1 for item in sublist]
#print(30 == len(set(t2)))
#print(len(set(t2)))

circles["children"].append({"name":"languages", "children":languages})



import json
with open("js/flare.json","w") as f:
  json.dump(circles, f)

